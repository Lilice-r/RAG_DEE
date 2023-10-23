#!/usr/bin/env python
# encoding:utf-8

from functools import reduce

import datasets
import random
import torch
import transformers
from transformers import DataCollatorForSeq2Seq
from sentence_retrieve.semantic_retrieve import SemanticRetrieve
from util.file_util import FileUtil


class T5DEEDataLoader(object):
    """
    t5 DEE模型数据加载类
    """
    def __init__(self, args, t5_model, t5_tokenizer, accelerator, logger):
        self.args = args
        self.t5_model = t5_model
        self.t5_tokenizer = t5_tokenizer
        self.accelerator = accelerator
        self.logger = logger
        self.sem_retrieve = SemanticRetrieve(self.args)

    def get_same_element_index(self, ob_list, word):
        return [i for (i, v) in enumerate(ob_list) if v == word]

    def schema_length(self, batch_ids):
        sep_id = self.t5_tokenizer.convert_tokens_to_ids('[SEP]')
        schema_lens = list()
        for input_id in batch_ids:
            sep_loc = self.get_same_element_index(input_id, sep_id)
            if sep_loc:
                schema_len = len(input_id[:sep_loc[0]])
            schema_lens.append(schema_len)
        return schema_lens

    def load_data(self, data_path, demo_path, batch_size, is_train=False) -> torch.utils.data.DataLoader:
        """
        加载数据
        :param data_path:
        :param batch_size:
        :param is_train:
        :return:
        """
        def tokenize_batch_func(batch_items):
            """
            处理批量训练数据
            :param batch_items: 一个batch的数据
            :batch["source_text"]: scr [SEP] event schema [SEP] demo1 <IN_SEP> demo2 <IN_SEP> demo3
            :return:
            """
            # 构造输入数据: bs, len
            model_inputs = self.t5_tokenizer(
                batch_items["source_text"],
                max_length=self.args.max_input_len,
                padding="max_length" if self.args.pad_to_max_length else False,
                truncation=True,
                return_tensors='np',
                # 已经预处理好为word list
                is_split_into_words=True
            )

            # 构造离散demo bsxK, len
            if self.args.add_demo:
                demo_inputs = self.t5_tokenizer(
                    sum(batch_items["discrete_demos"], []),
                    max_length=self.args.max_demo_len,
                    padding="max_length" if self.args.pad_to_max_length else False,
                    truncation=True,
                    return_tensors='np',
                    # 已经预处理好为word list
                    is_split_into_words=True
                )

                # 存储segment length

                schema_lens = self.schema_length(model_inputs["input_ids"])

            # 构造标签数据
            batch_label_list = []
            for word_label in batch_items["target_text"]:
                batch_label_list.append(word_label)

            # 切分输出句子
            with self.t5_tokenizer.as_target_tokenizer():
                labels = self.t5_tokenizer(
                    batch_label_list,
                    max_length=self.args.max_target_len,
                    padding="max_length" if self.args.pad_to_max_length else False,
                    truncation=True,
                    # 已经预处理好为word list
                    is_split_into_words=True
                )

            # 将padding对应的token替换为-100，则后续loss将忽略对应的padding token
            if self.args.pad_to_max_length and self.args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(label_id if label_id != self.t5_tokenizer.pad_token_id else -100) for label_id in label_ids]
                    for label_ids in labels["input_ids"]
                ]
            bs = model_inputs["input_ids"].shape[0]
            model_inputs["labels"] = labels["input_ids"]
            if self.args.add_demo:
                model_inputs["demos_id"] = demo_inputs["input_ids"].reshape((bs, -1))    # bs, Kxlen
                model_inputs["demos_attention_mask"] = demo_inputs["attention_mask"].reshape((bs, -1))
                model_inputs["schema_lens"] = schema_lens
            model_inputs["doc_id"] = batch_items["doc_id"]
            # model_inputs["tri_offset"] = batch_items["tri_offset"]

            return model_inputs

        # 加载原始数据
        # 如果添加demo, 则将data_path替换为拼接完样例的输入形式

        if self.args.add_demo:
            input_list = self.sem_retrieve.get_sim_sent(data_path, self.args.train_data_path, is_train)
            FileUtil.save_json_data(input_list, demo_path)
            print("accomplished single demo retrieval...")
            data_path = demo_path

        dee_dataset = datasets.load_dataset("json", data_files=data_path, split="train")
        print("loading success...")
        with self.accelerator.main_process_first():
            # 切分token，同时构造golden输出
            dee_dataset = dee_dataset.map(tokenize_batch_func, batched=True, batch_size=1000,
                                          remove_columns=dee_dataset.column_names)
                                          # num_proc=self.args.dataloader_proc_num)

        # 打印部分数据供观察
        # for index in random.sample(range(len(dee_dataset)), 3):
        #     self.logger.info(f"Sample {index} of the dataset: {dee_dataset[index]}\n")
        #
        data_collator = DataCollatorForSeq2Seq(
            self.t5_tokenizer,
            model=self.t5_model,
            label_pad_token_id=-100 if self.args.ignore_pad_token_for_loss else self.t5_tokenizer.pad_token_id,
        )

        dataloader = torch.utils.data.DataLoader(
            dee_dataset, shuffle=is_train, collate_fn=data_collator, batch_size=batch_size)



        # 解码打印观察数据
        # for index, batch_data in enumerate(dataloader):
        #     batch_data["labels"] = [
        #         [label_id for label_id in label_ids if label_id != -100]
        #         for label_ids in batch_data["labels"]
        #     ]
        #     input = self.t5_tokenizer.decode(batch_data["input_ids"][0])
        #     label = self.t5_tokenizer.decode(batch_data["labels"][0])
        #     print(f"input: {input}------------label: {label}\n")
        #     if index > 20:
        #         break

        return dataloader

