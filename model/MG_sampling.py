import torch


class MGSampling(object):
    """
    混合高斯采样->连续语义向量
    :return
    """
    def __init__(self):
        self.trunc_normal = torch.nn.init.trunc_normal_

    def mgrc_sampling(self, src_embedding, e_embedding, d_embedding):   # bsxK,768
        r = torch.norm(src_embedding - d_embedding, dim=1)     # bsxK
        R = torch.norm(src_embedding - e_embedding, dim=1)  # bsxK

        bias_vector = src_embedding - e_embedding   # bsxK,768

        W_r = (torch.abs(bias_vector) - torch.min(torch.abs(bias_vector), dim=1, keepdim=True).values) / \
              (torch.max(torch.abs(bias_vector), 1, keepdim=True).values - torch.min(torch.abs(bias_vector), 1, keepdim=True).values)   # bsxK,768
        mean = (2 - torch.div(r, R)) / 2          # bsxK
        omega = torch.randn_like(bias_vector) * W_r + mean.unsqueeze(1)
        sample = e_embedding + torch.mul(omega, bias_vector)

        return sample  # bsxK,768
