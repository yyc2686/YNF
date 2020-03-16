# -*- coding: utf-8 -*-

"""
概要：将python日志功能的常用配置封装成一个类LoggingPython

    :param
        log_name, default: 'untitled_log'
        save_path, default: DIR_dict['LOG_DIR']
    :return:
        logger

使用方法：

    步骤1：实例化LoggingPython，示例：client = LoggingPython(log_name='test')
    步骤2：使用logger的info, warning, error方法，示例：
            client.logger.info('Hello')
            >> 2020-02-01 20:53:27,648 test- INFO - Hello
            client.logger.warning('Hello')
            >> 2020-02-01 20:53:27,648 test- WARNING - Hello
            client.logger.error('Hello')
            >> 2020-02-01 20:53:27,649 test- ERROR - Hello
"""

import os
import logging
from data.config import DIR_dict


class LoggingPython(object):

    def __init__(self, **kwargs):
        self.log_name = kwargs.get("log_name", 'untitled_log')
        self.save_path = kwargs.get("save_path", DIR_dict['LOG_DIR'])
        os.makedirs(self.save_path, exist_ok=True)
        self.logger = self.logging_python()

    def logging_python(self):
        logger = logging.getLogger(self.log_name)  # 定义对应的程序模块名name，默认是root
        logger.setLevel(logging.INFO)  # 指定最低的日志级别 critical > error > warning > info > debug

        consol_haddler = logging.StreamHandler()  # 日志输出到屏幕控制台
        consol_haddler.setLevel(logging.INFO)  # 设置日志等级

        #  这里进行判断，如果logger.handlers列表为空，则添加，否则，直接去写日志，解决重复打印的问题
        if not logger.handlers:
            file_haddler = logging.FileHandler(self.save_path + '\\' + "{0}.log".format(self.log_name),
                                               encoding="utf-8")
            # 向文件log.txt输出日志信息，encoding="utf-8",防止输出log文件中文乱码
            file_haddler.setLevel(logging.INFO)  # 设置输出到文件最低日志级别

            formatter = logging.Formatter("%(asctime)s %(name)s- %(levelname)s - %(message)s")

            consol_haddler.setFormatter(formatter)  # 选择一个输出格式，可以定义多个输出格式
            file_haddler.setFormatter(formatter)

            logger.addHandler(file_haddler)  # 增加指定的handler
            logger.addHandler(consol_haddler)
        return logger


if __name__ == '__main__':
    # 示例
    # client = LoggingPython(log_name='test')
    # client.logger.info('Hello')
    # client.logger.warning('Hello')
    # client.logger.error('Hello')
    pass
