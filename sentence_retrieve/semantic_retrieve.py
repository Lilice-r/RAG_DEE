#!/usr/bin/env python
# encoding:utf-8

import sys
sys.path.append("../../RAG_DEE")
import torch
import sentence_transformers
import itertools
from util.file_util import FileUtil
from bean.dee_data_bean import DEEDataBean
from time import time
from datetime import timedelta
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SemanticRetrieve(object):
    """
    语义检索
    """
    def __init__(self, args):
        self.sent_model = sentence_transformers.SentenceTransformer(args.pre_train_model_path)
        self.soft = torch.nn.Softmax(dim=1)
        self.args = args
        self.k = args.k_demos

    def get_sim_sent(self, query_data_path, doc_data_path, is_train):
        """
        获取每条句子最相似的句子
        :param query_data_path:
        :param doc_data_path:
        :return: input: scr [SEP] event schema [SEP] demo1 <IN_SEP> demo2 <IN_SEP> demo3
        """
        query_data_list = FileUtil.read_json_data(query_data_path)
        doc_data_list = FileUtil.read_json_data(doc_data_path)

        all_doc_id = [DEEDataBean(**item).doc_id for item in query_data_list]
        oral_query_list = [" ".join(DEEDataBean(**item).source_text) for item in query_data_list]

        # setting 1 & 3
        if self.args.setting_type == 'contextual_semantic':
            all_query_list = oral_query_list
            all_doc_list = [" ".join(DEEDataBean(**item).tagged_source_text) for item in doc_data_list]

        # setting 2
        elif self.args.setting_type == 'event_pattern':
            all_doc_list = [" ".join(DEEDataBean(**item).target_text) for item in doc_data_list]
            if is_train:
                all_query_list = [" ".join(DEEDataBean(**item).target_text) for item in query_data_list]
            else:
                all_query_list = [" ".join(sum(DEEDataBean(**item).role_type, [])) for item in query_data_list]

        all_query_label = [DEEDataBean(**item).target_text for item in query_data_list]
        all_event_type = [DEEDataBean(**item).event_type for item in query_data_list]
        all_role_type = [DEEDataBean(**item).role_type for item in query_data_list]
        all_tri_offset = [DEEDataBean(**item).tri_offset for item in query_data_list]
        all_query_emds = self.sent_model.encode(all_query_list, convert_to_tensor=True)

        all_doc_labels = [" ".join(DEEDataBean(**item).target_text) for item in doc_data_list]  # 仅标签
        all_doc_event_type = [DEEDataBean(**item).event_type for item in doc_data_list]
        all_doc_emds = self.sent_model.encode(all_doc_list, convert_to_tensor=True)     # 仅源文本，无标签



        retrieval_list = []
        num = 0
        for id, query_sent, query_embedding, query_label, event_type, role_type, tri_offset in zip(all_doc_id, oral_query_list, all_query_emds, all_query_label, all_event_type, all_role_type, all_tri_offset):
            cos_scores = sentence_transformers.util.pytorch_cos_sim(query_embedding, all_doc_emds)[0]
            top_results = torch.topk(cos_scores, k=self.k+1)
            candi_demo_set, candi_tagged_demo_set = list(), list()

            if is_train:
                top_scores, top_idxes = top_results[0][1:], top_results[1][1:]
            else:
                top_scores, top_idxes = top_results[0][:-1], top_results[1][:-1]

            for score, idx in zip(top_scores, top_idxes):

                demo = all_doc_list[idx]    # str
                candi_demo_set.append(demo.split())
                if event_type == all_doc_event_type[idx]:
                    num += 1
                score = str(round(score.item(), 2))
            final_demo_set = candi_demo_set
            event_schema = list(itertools.chain.from_iterable(zip(event_type, role_type)))
            new_input = list(set(sum(event_schema, []))) + ['[SEP]'] + query_sent.split(" ")

            retrieval_list.append({"doc_id": id, "source_text": new_input[:-1], "target_text": query_label, "discrete_demos": final_demo_set, "tri_offset": tri_offset})

        return retrieval_list



def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))




if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    dataset = 'rams'

    pre_train_model_path = "/data/renyubing/project_data/pre_train_model/all-mpnet-base-v2/"
    train_path = "./data/" + dataset + "/prepro_train.jsonlines"
    dev_path = "./data/" + dataset + "/prepro_dev.jsonlines"
    test_path = "./data/" + dataset + "/prepro_test.jsonlines"

    query_path = dev_path
    doc_path = train_path


    start_time = time()
    sem_retrieve = SemanticRetrieve(pre_train_model_path)
    data_with_demo = sem_retrieve.get_sim_sent(query_path, doc_path)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)



