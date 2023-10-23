import re
import torch


class TextUtil(object):
    """
    文本处理工具类
    """

    @staticmethod
    def truecase_sentence(token_list):
        """
        将全大写英文字符转化为首字母大写的字符
        :param token_list:
        :return:
        """
        new_token_list = token_list[:]
        # 仅转换均为英文的token
        en_idx_token_list = [(idx, token) for idx, token in enumerate(token_list) if all(c.isalpha() for c in token)]
        en_token_list = [token for _, token in en_idx_token_list if re.match(r'\b[A-Z\.\-]+\b', token)]

        if len(en_token_list) and len(en_token_list) == len(en_idx_token_list):
            case_token_list = truecase.get_true_case(' '.join(en_token_list)).split()

            if len(case_token_list) == len(en_token_list):
                for (idx, token), case_token in zip(en_idx_token_list, case_token_list):
                    new_token_list[idx] = case_token
                return new_token_list

        return token_list

    @staticmethod
    def segment_embeds(segment_lens, batch_embeds):
        """
        将每段嵌入取出
        :param segment_lens: bs,6
        :param batch_embeds: bs,seq_len,hs
        :return: bs,hs  bs,hs   bs,3,hs
        """
        src_embedding = list()
        schema_embeddings = list()
        demo_embeddings = list()
        for len, embeds in zip(segment_lens, batch_embeds):
            src_end = len[0]
            schema_end = src_end + len[1]
            demo1_end = schema_end + len[2]
            demo2_end = demo1_end + len[3]
            demo3_end = demo2_end + len[4]
            src_embeds = embeds[:src_end].mean(dim=0)
            schema_embeds = embeds[src_end: schema_end].mean(dim=0)
            demo1_embeds = embeds[schema_end: demo1_end].mean(dim=0)
            demo2_embeds = embeds[demo1_end: demo2_end].mean(dim=0)
            demo3_embeds = embeds[demo2_end: demo3_end].mean(dim=0)
            src_embedding.append(src_embeds)
            schema_embeddings.append(schema_embeds)
            demo_embeddings.append(torch.stack([demo1_embeds, demo2_embeds, demo3_embeds], dim=0))
        return torch.stack(src_embedding, dim=0), torch.stack(schema_embeddings, dim=0), torch.stack(demo_embeddings, dim=0)
