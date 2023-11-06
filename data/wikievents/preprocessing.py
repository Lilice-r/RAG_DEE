from util.file_util import FileUtil
import json
from transformers import AutoTokenizer


class PREPRO:
    def __init__(self, data_path):
        self.data_list = FileUtil.read_json_data(data_path)

    def initial_prepro(self, out_path):
        """把原始数据初步处理成易读的格式"""
        raw_list = self.data_list
        initial_list = list()
        id = 0
        for doc in raw_list:
            entitys_dict = {}
            for entity in doc['entity_mentions']:
                entitys_dict[entity['id']] = [entity['start'], entity['end']]
            events_list = doc['event_mentions']
            events_list_su = list()
            for event in events_list:
                type = event['event_type']
                trigger = [event['trigger']['start'], event['trigger']['end']]
                args_list = list()
                for arg in event['arguments']:
                    role = arg['role']              # 角色
                    arg_id = arg['entity_id']
                    offset = entitys_dict[arg_id]   # [start, end]
                    args_list.append([role, offset])
                if args_list:
                    events_list_su.append({"type": type, "tokens": trigger, "arguments": args_list})
            if events_list_su:
                initial_list.append({"doc_id": id, "text": doc['tokens'], "event": events_list_su})
                id += 1
        FileUtil.save_json_data(initial_list, out_path)
        return initial_list

    def get_same_element_index(self, ob_list, word):
        return [i for (i, v) in enumerate(ob_list) if v == word]

    def constrain_max_length(self, out_path):
        t5_tokenizer = AutoTokenizer.from_pretrained("/data/renyubing/project_data/pre_train_model/t5-base")
        t5_tokenizer.add_tokens(['<tgr>', '<IN_SEP>', '[SEP]'])
        sep_id = t5_tokenizer.convert_tokens_to_ids('[SEP]')
        prepro_data = self.data_list
        new_data = list()
        for data in prepro_data:
            src_text = data["source_text"]
            seq_id = t5_tokenizer(src_text, is_split_into_words=True)["input_ids"]
            seq_len = len(seq_id)
            if seq_len <= 512:
                source_text = src_text
            else:
                sep_loc = self.get_same_element_index(seq_id, sep_id)
                concat_token = t5_tokenizer.decode(seq_id[sep_loc[0]:], skip_special_tokens=True)
                concat_len = len(seq_id[sep_loc[0]:])
                src_len = 512 - concat_len
                source_text = t5_tokenizer.decode(seq_id[:src_len], skip_special_tokens=True) + concat_token
                source_text = source_text.replace(",", " ,")
                source_text = source_text.replace("[SEP]", " [SEP]")
                source_text = source_text.replace("<tgr>", " <tgr>")
                source_text = source_text.replace("<IN_SEP>", " <IN_SEP>").split()
                print(src_len, concat_len, len(source_text), len(t5_tokenizer(source_text, is_split_into_words=True)["input_ids"]), data["doc_id"])
            new_data.append({"doc_id": data["doc_id"], "source_text": source_text, "target_text": data["target_text"], "event_type": data["event_type"], "role_type": data["role_type"], "tri_offset": data["tri_offset"]})
        FileUtil.save_json_data(new_data, out_path)

    def extract_to_generate(self, out_path):
        raw_list = self.data_list   # {text:[], event:[{type: , tokens: [], arguments: [[role, [start, end]], ...]}, ...]
        gene_list = list()
        event2role = dict()
        with open('./event_role_multiplicities.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split()
                event2role[line[0]] = line[1:]
        for sentence in raw_list:
            if len(sentence['event']) == 0:
                continue
            text_list = sentence['text']
            target_list = []
            role_type = []
            event_type = []
            tri_offsets = []
            trigger = []
            for e in sentence['event']:
                tri_start = e['tokens'][0]
                tri_end = e['tokens'][1]
                # tri = ' '.join(text_list[tri_start: (tri_end)])
                # target_list += [tri, '[' + e['type'] + ']']
                for arg in e['arguments']:
                    role = arg[0]
                    arg_start = arg[1][0]
                    arg_end = arg[1][1]
                    arg = ' '.join(text_list[arg_start: arg_end])
                    target_list += text_list[arg_start: arg_end] + ['[' + role + ']']
                target_list.append('[SEP]')
                event_type.append([e['type'].split('.')[0]])
                trigger.append(e['tokens'])
                role_type.append(event2role[e['type']])
                tri_offsets += [tri_start]
            new_text = list()
            for idx, token in enumerate(text_list):
                if idx in tri_offsets:
                    new_text += ['<tgr>', token, '<tgr>']
                else:
                    new_text += [token]
            gene_list.append({'doc_id': sentence['doc_id'], 'source_text': new_text, 'tagged_source_text': new_text, 'target_text': target_list[:-1], 'event_type': event_type, 'role_type': role_type, 'tri_offset': trigger})
        FileUtil.save_json_data(gene_list, out_path)

# PREPRO('./train.jsonl').initial_prepro('/data/renyubing/project_data/ee/RAG_DEE/WIKIEVENTS/train_init.jsonl')
PREPRO('/data/renyubing/project_data/ee/RAG_DEE/WIKIEVENTS/train_init.jsonl').extract_to_generate('/data/renyubing/project_data/ee/RAG_DEE/WIKIEVENTS/prepro_train.jsonl')
# PREPRO("/data/renyubing/project_data/ee/RAG_DEE/WIKIEVENTS/dev_with_3sbdemo.jsonl").constrain_max_length("/data/renyubing/project_data/ee/RAG_DEE/WIKIEVENTS/dev_with_3sbdemo_constrain512.jsonl")

