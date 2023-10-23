import torch
from dataclasses import dataclass, field


@dataclass
class BaseArgBean(object):
    """
    参数定义基类
    """
    # 数据文件相关
    train_data_path: str = field(default=None)
    dev_data_path: str = field(default=None)
    test_data_path: str = field(default=None)
    train_demo_path: str = field(default=None)
    dev_demo_path: str = field(default=None)
    test_demo_path: str = field(default=None)
    model_save_path: str = field(default=None)


    # -->
    retrieval_train_path: str = field(default=None)
    retrieval_dev_path: str = field(default=None)
    retrieval_test_path: str = field(default=None)
    # -->

    pretrain_model_path: str = field(default=None)
    pre_train_model_path: str = field(default=None)
    gptneo_model_path: str = field(default=None)
    output_dir: str = field(default=None)

    # 状态相关
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    do_predict: bool = field(default=False)
    add_vec: bool = field(default=False)
    add_demo: bool = field(default=True)
    setting_type: str = field(default='contextual_semantic')
    dataset: str = field(default='rams')


    # 模型参数相关
    dataloader_proc_num: int = field(default=4)
    epoch_num: int = field(default=5)
    eval_batch_step: int = field(default=2)
    require_improvement_step: int = field(default=1000)
    num_processes: int = field(default=2)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)

    # 超过这个长度将被截断
    max_input_len: int = field(default=512)
    max_demo_len: int = field(default=150)
    pad_to_max_length: bool = field(default=False)
    learning_rate: float = field(default=0.02)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed: int = field(default=42)

    @property
    def train_batch_size(self) -> int:
        """
        训练batch_size，多卡训练时为per_device_train_batch_size*device_count
        当前只考虑了GPU情况
        """
        train_batch_size = self.per_device_train_batch_size * max(1, torch.cuda.device_count())
        return train_batch_size

    @property
    def test_batch_size(self) -> int:
        """
        训练batch_size，多卡训练时为per_device_train_batch_size*device_count
        当前只考虑了GPU情况
        """
        test_batch_size = self.per_device_eval_batch_size * max(1, torch.cuda.device_count())
        return test_batch_size


@dataclass
class BartArgBean(BaseArgBean):
    """
    BART模型参数定义类
    """
    # 模型参数相关
    max_target_len: int = field(default=512)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.1)
    beam_num: int = field(default=1)
    ignore_pad_token_for_loss: bool = field(default=True)
    event_type_path: str = field(default=None)
    role_type_path: str = field(default=None)


@dataclass
class T5ArgBean(BaseArgBean):
    """
    T5模型参数定义类
    """
    # 模型参数相关
    max_target_len: int = field(default=512)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.1)
    accumulate_grad_batches: int = field(default=2)
    beam_num: int = field(default=1)
    ignore_pad_token_for_loss: bool = field(default=True)
    event_type_path: str = field(default=None)
    role_type_path: str = field(default=None)
    k_demos: int = field(default=5)