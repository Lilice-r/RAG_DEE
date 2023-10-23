#!/usr/bin/env python
# encoding:utf-8

import torch
import numpy as np
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from model.dee_metric import DEEMtric


class T5DEEProcess(object):
    """
    基于t5生成模型的DEE
    """

    def __init__(self, args, t5_model, t5_tokenizer, accelerator, logger):
        self.args = args
        self.t5_model = t5_model
        self.t5_tokenizer = t5_tokenizer
        self.accelerator = accelerator
        self.logger = logger
        self.dee_metric = DEEMtric(args)
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def train(self, train_loader, dev_loader):
        """
        :param train_loader:
        :param dev_loader:
        :return:
        """
        self.t5_model.train()

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.t5_model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.t5_model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        t_total = len(train_loader) * self.args.epoch_num
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * self.args.warmup_ratio),
                                                    num_training_steps=t_total)

        # Prepare everything with "accelerator", 会统一将模型、数据加载到对应的device
        self.t5_model, optimizer, train_loader, dev_loader = self.accelerator.prepare(
            self.t5_model, optimizer, train_loader, dev_loader)

        # 进行到多少batch
        total_batch = 0
        dev_best_score_arg = float("-inf")
        # 上次验证集loss下降的batch数
        last_improve = 0
        # 是否很久没有效果提升
        no_improve_flag = False

        self.logger.info(f"Total Train Batch Num: {len(train_loader)}")
        for epoch in range(self.args.epoch_num):
            self.logger.info(f"Epoch [{epoch + 1}/{self.args.epoch_num}]")
            for step, batch_data in enumerate(train_loader):
                self.logger.info(step)

                if self.args.add_demo:
                    outputs = self.t5_model(input_ids=batch_data["input_ids"], attention_mask=batch_data["attention_mask"], demos_id=batch_data["demos_id"], demos_attention_mask=batch_data["demos_attention_mask"], segment_lens=batch_data["schema_lens"], labels=batch_data["labels"])
                else:
                    outputs = self.t5_model(input_ids=batch_data["input_ids"], attention_mask=batch_data["attention_mask"], labels=batch_data["labels"])

                loss = outputs.loss
                self.accelerator.backward(loss)

                if ((step+1)%self.args.accumulate_grad_batches)==0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # 输出在验证集上的效果
                if total_batch % len(train_loader) == 0 and total_batch > 0:
                    eval_metric = self.evaluate(dev_loader)

                    if eval_metric["argC_f1"] > dev_best_score_arg:
                        dev_best_score_arg = eval_metric["argC_f1"]
                        self.accelerator.save(
                            self.accelerator.unwrap_model(self.t5_model).state_dict(), self.args.model_save_path)
                        improve = "*"
                        last_improve = total_batch
                    else:
                        improve = ""
                    self.logger.info(
                        f'Iter: {total_batch}, Train Loss: {loss.item()}, argI_eval_metric: {eval_metric["argI_precision"], eval_metric["argI_recall"], eval_metric["argI_f1"]}, argC_eval_metric: {eval_metric["argC_precision"], eval_metric["argC_recall"], eval_metric["argC_f1"]} {improve}, HeadC_eval_metric: {eval_metric.get("HeadC_f1", 0)}')

                    self.t5_model.train()



                total_batch += 1
                require_improvement_step = len(train_loader) * 5
                if self.args.dataset == 'wiki':
                    require_improvement_step = len(train_loader) * 10
                if total_batch - last_improve > require_improvement_step:
                    self.logger.info("No optimization for a long time, auto-stopping...")
                    no_improve_flag = True
                    break
            if no_improve_flag:
                break

    def evaluate(self, dev_loader):
        """
        验证模型
        :param dev_loader:
        :return:
        """
        self.t5_model.eval()

        with torch.no_grad():
            for step, batch_data in enumerate(dev_loader):
                if self.args.add_demo:
                    batch_generate_ids = self.accelerator.unwrap_model(self.t5_model).generate(
                        batch_data["input_ids"],
                        attention_mask=batch_data["attention_mask"],
                        max_length=self.args.max_target_len,
                        num_beams=self.args.beam_num,
                        decoder_segment_lens=batch_data["schema_lens"],
                        decoder_encoder_input_ids=batch_data["input_ids"],
                        decoder_demos_id=batch_data["demos_id"],
                        decoder_demos_attention_mask=batch_data["demos_attention_mask"]
                    )
                else:
                    batch_generate_ids = self.accelerator.unwrap_model(self.t5_model).generate(
                        batch_data["input_ids"],
                        max_length=self.args.max_target_len,
                        num_beams=self.args.beam_num
                    )

                # 不同进程生成的长度不一致因此需要padding
                batch_generate_ids = self.accelerator.pad_across_processes(
                    batch_generate_ids, dim=1, pad_index=self.t5_tokenizer.pad_token_id)

                batch_label_ids = batch_data["labels"]
                if not self.args.pad_to_max_length:
                    # 将多个不同设备上的tensor padding到同一维度(当使用的动态padding时, 不同设备最大长度会不一致)
                    batch_label_ids = self.accelerator.pad_across_processes(
                        batch_label_ids, dim=1, pad_index=self.t5_tokenizer.pad_token_id)

                # 聚合多个不同设备的tensor并将其拼接
                generate_ids_gathered = self.accelerator.gather(batch_generate_ids).cpu().clone().numpy()
                label_ids_gathered = self.accelerator.gather(batch_label_ids).cpu().clone().numpy()

                # 将id解码为对应的token
                if self.args.ignore_pad_token_for_loss:
                    # 将label中的-100替换为tokenizer中的pad_id，否则无法解码
                    label_ids_gathered = np.where(
                        label_ids_gathered != -100, label_ids_gathered, self.t5_tokenizer.pad_token_id)

                input_tokens = self.t5_tokenizer.batch_decode(batch_data["input_ids"], skip_special_tokens=True)
                generate_tokens = self.t5_tokenizer.batch_decode(generate_ids_gathered, skip_special_tokens=True)
                label_tokens = self.t5_tokenizer.batch_decode(label_ids_gathered, skip_special_tokens=True)


                print(f'input: {input_tokens[0]}, generate_result: {generate_tokens[0]}, label: {label_tokens[0]}')

                # 将当前batch结果加入评测
                self.dee_metric.add_generate_batch(batch_doc_ids=batch_data['doc_id'], batch_pred_tokens=generate_tokens)

        # 计算评测指标
        print(self.dee_metric.static_dict)
        eval_metric = self.dee_metric.compute_generate_metric()


        return eval_metric

    def test(self, test_loader):
        """
        测试模型
        :param test_loader:
        :return:
        """
        # 加载模型
        self.t5_model.load_state_dict(torch.load(self.args.model_save_path))
        self.t5_model.eval()
        self.t5_model, test_loader = self.accelerator.prepare(self.t5_model, test_loader)

        test_metric = self.evaluate(test_loader)
        self.logger.info(f'argI_test_metric: {test_metric["argI_precision"], test_metric["argI_recall"], test_metric["argI_f1"]}, argC_test_metric: {test_metric["argC_precision"], test_metric["argC_recall"], test_metric["argC_f1"]}, HeadC_test_metric: {test_metric.get("HeadC_f1", 0)}')

