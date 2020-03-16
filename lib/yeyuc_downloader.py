# -*- coding: utf-8 -*-

"""
概要：将you_get的常用配置封装成一个类YouGet

使用方法：

    步骤1：实例化YouGet，示例：client = YouGet()
    步骤2：配置path, urls
    步骤3：调用download进行下载

"""

import os
import sys
import time

import you_get

from data.config import BASE_DIR
from lib.common import Common
from lib.yeyuc_multicore import MultiCore


class YouGet():

    def __init__(self, **kwargs):
        self.path = kwargs.get('path') if kwargs.get('path') else self.save_path()
        self.common = Common()

    def save_path(self):
        parallel = os.path.abspath(os.path.dirname(BASE_DIR))
        flv_path = self.common.mkdir(os.path.join(parallel, 'you_get'))
        return flv_path

    def download(self, url):
        sys.argv = ['you-get', '-o', self.path, url]
        you_get.main()

        # 下载完成，删除xml文件
        for file in os.listdir(self.path):
            if file[-3:] == 'xml':
                self.common.rmfile(os.path.join(self.path, file))


class Thunder():

    def __init__(self, **kwargs):
        self.common = Common()
        self.path = kwargs.get('path') if kwargs.get('path') else self.save_path()

        from win32com.client import Dispatch
        self.thunder = Dispatch('ThunderAgent.Agent64.1')

    def save_path(self):
        parallel = os.path.abspath(os.path.dirname(BASE_DIR))
        flv_path = self.common.mkdir(os.path.join(parallel, 'you_get'))
        return flv_path

    def download1(self, urls):
        for i, url in enumerate(urls):
            self.thunder.AddTask(url, "第{0}集.rmvb".format(i + 1), self.path)
            self.thunder.CommitTasks()

    def download(self, url, name):
        self.thunder.AddTask(url, "第{0}集.rmvb".format(name))
        self.thunder.CommitTasks()
        time.sleep(60)


class Work(MultiCore):

    def job(self, data):
        urls = ["https://www.bilibili.com/video/av23740395?p={0}".format(i) for i in range(data[0], data[1])]
        path = r'E:\共享文件夹\后端之路\后端基础\计算机操作系统\南京大学'

        client = YouGet(path=path)
        for url in urls:
            client.download(url)


if __name__ == '__main__':
    path = r"E:\共享文件夹\杂物\huoying"
    # url = "https://v.youku.com/v_show/id_XNTI4NjExNTMy.html?spm=a2hbt.13141534.app.5~5!2~5!2~5~5~5!2~5~5!2~5!2~5!2~5~A!5"
    url = "https://v.youku.com/v_show/id_XNTI4NjEwODg4.html?spm=a2hbt.13141534.app.5~5!2~5!2~5~5~5!2~5~5!2~5!2~5!2~5~A!"
    client = YouGet(path=path)
    for i in range(1, 51):
        client.download(url + str(i))

    # path = r"E:\共享文件夹\杂物\huoying"
    # urls = [
    #     "thunder://QUFmdHA6Ly95Z2R5ODp5Z2R5OEB5Zzc2LmR5ZHl0dC5uZXQ6MzUyNi9b0fS54rXn07Atd3d3LnlnZHk4LmNvbV278NOwyMzV3y03MjAubXA0Wlo=",
    #     "thunder://QUFmdHA6Ly95Z2R5ODp5Z2R5OEB5Zzc2LmR5ZHl0dC5uZXQ6MzUyMC9b0fS54rXn07Atd3d3LnlnZHk4LmNvbV278NOwyMzV3y03MTkubWt2Wlo="]
    #
    # client = Thunder(path=path)
    # client.download(urls)

    # client = Work()
    # n = 95
    # data = [(10 * i + 1, 10 * (i + 1)) for i in range(int(n / 10))] + [(10 * int(n / 10) + 1, n + 1)]
    # # data = [(10 * i, 10 * i + 1) for i in range(1, int(n / 10) + 1)]
    # # data = [(81, 82)]
    # client.process(data)
