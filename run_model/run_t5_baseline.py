#!/usr/bin/env python
# encoding:utf-8

import sys
sys.path.append("../RAG_DEE")
import os
import logging
import torch
import transformers
from accelerate import Accelerator
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM

from bean.arg_bean import T5ArgBean
from model.t5_baseline.t5_dee_dataloader import T5DEEDataLoader
from model.t5_baseline.t5_dee_process import T5DEEProcess
from model.t5_baseline.t5_dee_network import T5ForConditionalGen
from util.arg_parse import CustomArgParser
from util.custom_logger import CustomLogger

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# device_ids = [0, 1, 2, 3]

class T5DEEController(object):
    """
    BERT baseline DEE模型
    """
    def __init__(self, args):
        self.args = args
        print(self.args)
        self.args.t5_config = AutoConfig.from_pretrained(self.args.pretrain_model_path)
        self.t5_tokenizer = AutoTokenizer.from_pretrained(self.args.pretrain_model_path)
        if args.add_demo:
            self.t5_dee_model = T5ForConditionalGen.from_pretrained(self.args.pretrain_model_path, args)
        else:
            self.t5_dee_model = AutoModelForSeq2SeqLM.from_pretrained(self.args.pretrain_model_path)
        self.t5_tokenizer.add_tokens(['<tgr>', '<IN_SEP>', '[SEP]'])

        self.t5_dee_model.resize_token_embeddings(len(self.t5_tokenizer))

        # self.t5_dee_model = torch.nn.DataParallel(self.t5_dee_model, device_ids=device_ids)

        self.accelerator = Accelerator()
        self.logger = CustomLogger.logger
        # 多个进程仅打印1次log
        self.logger.setLevel(logging.INFO if self.accelerator.is_local_main_process else logging.ERROR)

        self.t5_dee_dataloader = T5DEEDataLoader(
            self.args, self.t5_dee_model, self.t5_tokenizer, self.accelerator, self.logger)
        self.t5_dee_process = T5DEEProcess(
            self.args, self.t5_dee_model, self.t5_tokenizer, self.accelerator, self.logger)

    def train(self):
        """
        训练模型
        :return:
        """

        # 加载数据，返回DataLoader
        self.logger.info(self.accelerator.state)
        self.logger.info("Loading data...")
        train_dataloader = self.t5_dee_dataloader.load_data(
            self.args.train_data_path, self.args.train_demo_path, self.args.per_device_train_batch_size, is_train=True)
        dev_dataloader = self.t5_dee_dataloader.load_data(
            self.args.dev_data_path, self.args.dev_demo_path, self.args.per_device_eval_batch_size, is_train=False)
        self.logger.info("Finished loading data ...")


        # 训练模型
        self.logger.info("Training model...")
        self.t5_dee_process.train(train_dataloader, dev_dataloader)
        self.logger.info("Finished Training model!!!")

    def test(self):
        """
        测试模型
        :return:
        """

        # 加载数据
        self.logger.info("Loading data...")
        test_dataloader = self.t5_dee_dataloader.load_data(
            self.args.test_data_path, self.args.test_demo_path, self.args.per_device_eval_batch_size, is_train=False)
        self.logger.info("Finished loading data!!!")

        # 测试模型
        self.logger.info("Testing model...")
        self.t5_dee_process.test(test_dataloader)


if __name__ == "__main__":
    # 解析命令行参数
    args = CustomArgParser(T5ArgBean).parse_args_into_dataclass()

    # 在初始化模型前固定种子，保证每次运行结果一致
    transformers.set_seed(args.seed)

    t5_dee_controller = T5DEEController(args)

    # 模型训练
    if args.do_train:
        t5_dee_controller.train()

    # 模型测试，有真实标签
    if args.do_eval:
        t5_dee_controller.test()

