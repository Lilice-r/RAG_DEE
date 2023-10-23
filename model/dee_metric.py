#!/usr/bin/env python
# encoding:utf-8

from util.file_util import FileUtil
from util.entity_util import EntityUtil, find_head
import spacy

nlp = spacy.load('en_core_web_sm')


def recall_gold_label(doc_id, data_path):
    files = FileUtil.read_json_data(data_path)
    argI_label = list()
    argC_label = list()
    HeadC_label = list()
    origin_input, tri_offset = list(), list()
    for line in files:
        if line['doc_id'] == doc_id:
            origin_input, event_label = line['text'], line['event']

    if not event_label:
        print(f"找不到{doc_id}")
    doc = nlp(' '.join(origin_input))

    for event in event_label:
        tri_offset.append(event["tokens"])
        argI_label += [arg[1] for arg in event["arguments"]]
        argC_label += [["["+arg[0]+"]", arg[1]] for arg in event["arguments"]]
        HeadC_label += [["["+arg[0]+"]"] + [find_head(arg[1][0], arg[1][1], doc)] for arg in event["arguments"]]
    return argI_label, argC_label, HeadC_label, origin_input, tri_offset


class DEEMtric(object):
    """
    模型处理基类
    """
    def __init__(self, args):
        self.args = args
        self.static_dict = {"pred_argI_num": 0, "label_argI_num": 0, "pred_right_argI_num": 0, "pred_argC_num": 0, "label_argC_num": 0, "pred_right_argC_num": 0}

    def add_generate_batch(self, batch_doc_ids=None, batch_pred_tokens=None):
        """
        添加生成模型中每个batch的结果对
        :param batch_input_tokens: 已解码为对应字符
        :param batch_pred_tokens: 已解码为对应字符
        :param batch_label_tokens: 已解码为对应字符
        :return:
        """
        pred_argI_num = 0
        label_argI_num = 0
        pred_right_argI_num = 0
        pred_argC_num = 0
        label_argC_num = 0
        pred_right_argC_num = 0
        pred_HeadC_num = 0
        label_HeadC_num = 0
        pred_right_HeadC_num = 0

        if self.args.dataset == 'wiki':
            dataset = 'WIKIEVENTS'
            postfix = 'jsonl'
        else:
            dataset = 'RAMS'
            postfix = 'jsonlines'

        with open(self.args.role_type_path, 'r', encoding='utf-8') as f:
            role_type = set(line.strip() for line in f.readlines())
        if self.args.do_train:
            eval_data_path = '/data/renyubing/project_data/ee/RAG_DEE/' + dataset + '/dev_init.' + postfix
        if self.args.do_eval:
            eval_data_path = '/data/renyubing/project_data/ee/RAG_DEE/' + dataset + '/test_init.' + postfix
        for doc_id, pred_tokens in zip(batch_doc_ids, batch_pred_tokens):
            label_argI_list, label_argC_list, label_HeadC_list, origin_input, tri_offset = recall_gold_label(doc_id, eval_data_path)
            pred_tokens = pred_tokens.replace("[SEP]", " [SEP]")
            pred_argI_list, pred_argC_list, pred_HeadC_list = EntityUtil.get_generate_entity(origin_input, pred_tokens.split(), role_type, tri_offset)
            print(f'Arg-I预测: {pred_argI_list}, Arg-I标签: {label_argI_list}, Arg-C预测: {pred_argC_list}, Arg-C标签: {label_argC_list}, H-C预测: {pred_HeadC_list}, H-C标签: {label_HeadC_list}')

            pred_argI_num += len(pred_argI_list)
            label_argI_num += len(label_argI_list)
            pred_right_argI_num += len([pred_argI for pred_argI in pred_argI_list if pred_argI in label_argI_list])

            pred_argC_num += len(pred_argC_list)
            label_argC_num += len(label_argC_list)
            pred_right_argC_num += len([pred_arg for pred_arg in pred_argC_list if pred_arg in label_argC_list])

            pred_HeadC_num += len(pred_HeadC_list)
            label_HeadC_num += len(label_HeadC_list)
            pred_right_HeadC_num += len([pred_Head for pred_Head in pred_HeadC_list if pred_Head in label_HeadC_list])

        self.static_dict["pred_argI_num"] = self.static_dict.get("pred_argI_num", 0) + pred_argI_num
        self.static_dict["label_argI_num"] = self.static_dict.get("label_argI_num", 0) + label_argI_num
        self.static_dict["pred_right_argI_num"] = self.static_dict.get("pred_right_argI_num", 0) + pred_right_argI_num
        self.static_dict["pred_argC_num"] = self.static_dict.get("pred_argC_num", 0) + pred_argC_num
        self.static_dict["label_argC_num"] = self.static_dict.get("label_argC_num", 0) + label_argC_num
        self.static_dict["pred_right_argC_num"] = self.static_dict.get("pred_right_argC_num", 0) + pred_right_argC_num

        self.static_dict["pred_HeadC_num"] = self.static_dict.get("pred_HeadC_num", 0) + pred_HeadC_num
        self.static_dict["label_HeadC_num"] = self.static_dict.get("label_HeadC_num", 0) + label_HeadC_num
        self.static_dict["pred_right_HeadC_num"] = self.static_dict.get("pred_right_HeadC_num", 0) + pred_right_HeadC_num

    def compute_generate_metric(self):
        """
        计算生成模型中所有batch的评测指标
        :return:
        """
        pred_argI_num = self.static_dict.get("pred_argI_num", 0)
        label_argI_num = self.static_dict.get("label_argI_num", 0)
        pred_right_argI_num = self.static_dict.get("pred_right_argI_num", 0)

        argI_precision = 0 if pred_argI_num == 0 else (pred_right_argI_num / pred_argI_num)
        argI_recall = 0 if label_argI_num == 0 else (pred_right_argI_num / label_argI_num)
        argI_f1 = 0 if argI_recall + argI_precision == 0 else (2 * argI_precision * argI_recall) / (argI_precision + argI_recall)

        pred_argC_num = self.static_dict.get("pred_argC_num", 0)
        label_argC_num = self.static_dict.get("label_argC_num", 0)
        pred_right_argC_num = self.static_dict.get("pred_right_argC_num", 0)

        argC_precision = 0 if pred_argC_num == 0 else (pred_right_argC_num / pred_argC_num)
        argC_recall = 0 if label_argC_num == 0 else (pred_right_argC_num / label_argC_num)
        argC_f1 = 0 if argC_recall + argC_precision == 0 else (2 * argC_precision * argC_recall) / (argC_precision + argC_recall)

        pred_HeadC_num = self.static_dict.get("pred_HeadC_num", 0)
        label_HeadC_num = self.static_dict.get("label_HeadC_num", 0)
        pred_right_HeadC_num = self.static_dict.get("pred_right_HeadC_num", 0)

        HeadC_precision = 0 if pred_HeadC_num == 0 else (pred_right_HeadC_num / pred_HeadC_num)
        HeadC_recall = 0 if label_HeadC_num == 0 else (pred_right_HeadC_num / label_HeadC_num)
        HeadC_f1 = 0 if HeadC_recall + HeadC_precision == 0 else (2 * HeadC_precision * HeadC_recall) / (HeadC_precision + HeadC_recall)

        # 重置统计结果，防止多次验证时重复计算
        self.static_dict = {"pred_argI_num": 0, "label_argI_num": 0, "pred_right_argI_num": 0}
        self.static_dict = {"pred_argC_num": 0, "label_argC_num": 0, "pred_right_argC_num": 0}
        self.static_dict = {"pred_HeadC_num": 0, "label_HeadC_num": 0, "pred_right_HeadC_num": 0}
        if self.args.dataset == 'wiki':
            return {"argI_precision": argI_precision, "argI_recall": argI_recall, "argI_f1": argI_f1, "argC_precision": argC_precision, "argC_recall": argC_recall, "argC_f1": argC_f1, "HeadC_f1": HeadC_f1}
        else:
            return {"argI_precision": argI_precision, "argI_recall": argI_recall, "argI_f1": argI_f1, "argC_precision": argC_precision, "argC_recall": argC_recall, "argC_f1": argC_f1}
