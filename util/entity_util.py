#!/usr/bin/env python
# encoding:utf-8
import sys
import spacy

nlp = spacy.load('en_core_web_sm')


def find_head(arg_start, arg_end, doc):  # doc = nlp(' '.join(context_words))
    arg_end -= 1
    cur_i = arg_start
    while doc[cur_i].head.i >= arg_start and doc[cur_i].head.i <= arg_end:
        if doc[cur_i].head.i == cur_i:
            # self is the head
            break
        else:
            cur_i = doc[cur_i].head.i

    arg_head = cur_i
    head_text = doc[arg_head]

    return str(head_text)


def find_arg_span(arg, context_words, trigger_start, trigger_end, head_only=False):
    match = None
    arg_len = len(arg)
    min_dis = len(context_words)  # minimum distance to trigger
    for i, w in enumerate(context_words):
        if context_words[i:i + arg_len] == arg:
            if i < trigger_start:
                dis = abs(trigger_start - i - arg_len)
            else:
                dis = abs(i - trigger_end)
            if dis < min_dis:
                match = [i, i + arg_len]
                min_dis = dis

    if match and head_only:
        doc = nlp(' '.join(context_words))
        assert (doc != None)
        match = find_head(match[0], match[1], doc)
    return match


class EntityUtil(object):
    """
    实体处理相关工具类
    """

    @staticmethod
    def get_seq_entity(token_labels) -> list:
        """
        根据序列标注结果获取实体
        :param token_labels: 序列标注 BIOS
        :return: ['B-PER', 'I-PER', 'O', 'S-LOC'] -> [['PER', 0, 1], ['LOC', 3, 3]]
        """
        entity_list = []
        entity = [-1, -1, -1]
        for index, tag in enumerate(token_labels):
            if tag.startswith("S-"):
                # 先前识别的实体
                if entity[2] != -1:
                    entity_list.append(entity)
                entity_list.append([tag.split("-")[1], index, index])
                entity = [-1, -1, -1]
            elif tag.startswith("B-"):
                # 先前识别的实体
                if entity[2] != -1:
                    entity_list.append(entity)
                entity = [tag.split("-")[1], index, -1]
            elif tag.startswith('I-') and entity[1] != -1:
                _type = tag.split('-')[1]
                if _type == entity[0]:
                    entity[2] = index
                if index == len(token_labels) - 1:
                    entity_list.append(entity)
            else:
                # 先前识别的实体
                if entity[0] != -1:
                    entity[2] = index - 1
                    entity_list.append(entity)
                entity = [-1, -1, -1]

        return entity_list

    @staticmethod
    def get_generate_entity(input_word_list: list, generate_word_list: list, role_label: set, tri_offsets: list):
        """
        根据生成模型结果获取实体列表
        :param generate_word_list: 生成的token序列
        :param label_set: 实体类型集合
        :return: ['Michael', 'Jordan', 'Person', 'Beijing', 'Location'] -> [['Person', 'Michael Jordan', [0,1]], ['Location', 'Beijing', []]]
        """
        name_list = []
        arg_I_list = []
        arg_C_list = []
        Head_C_list = []
        event_idx = 0
        for word in generate_word_list:
            if word != '[SEP]':
                if word not in role_label:
                    name_list.append(word)
                else:
                    matched_list = find_arg_span(name_list, input_word_list, tri_offsets[event_idx][0], tri_offsets[event_idx][1])
                    head_list = None
                    if matched_list:
                        head_list = find_head(matched_list[0], matched_list[1], nlp(' '.join(input_word_list)))
                    arg_C_list.append([word, matched_list])
                    arg_I_list.append(matched_list)
                    Head_C_list.append([word, head_list])
                    name_list = []
            else:
                if event_idx < len(tri_offsets)-1:
                    event_idx += 1
                name_list = []
        return arg_I_list, arg_C_list, Head_C_list