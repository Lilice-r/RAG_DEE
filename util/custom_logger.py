import logging


class CustomLogger(object):
    """
    日志工具类
    """
    # 日志格式
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(filename)s line:%(lineno)d] %(levelname)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger()

