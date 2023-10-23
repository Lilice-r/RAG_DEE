from dataclasses import dataclass


@dataclass()
class DEEDataBean(object):
    """
    DEE 数据格式定义
    """
    doc_id: str
    source_text: list
    tagged_source_text: list
    target_text: list
    event_type: list
    role_type: list
    tri_offset: list
