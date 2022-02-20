import os
import logging


def get_logger(filename):
    """
    获取logger对象
    :param filename:log文件路径
    :return:
    """
    path, _ = os.path.split(filename)
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel('INFO')
    basic_format = "%(asctime)s:%(levelname)s:%(message)s"
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(basic_format, date_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel('INFO')
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger
