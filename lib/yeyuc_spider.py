# -*- coding: utf-8 -*-

# requests_spider.py
# @author yeyuc
# @description
# @created 2020-01-29T08:58:07.023Z+08:00
# @last-modified 2020-01-31T17:28:00.856Z+08:00
#

"""
概要：北京理工大学嵩天教授团队Python网络爬虫与信息提取课程代码整理（静态爬虫）
技术路线：
    requests-re-bs4
爬虫限制：
    "URL"/robots.txt
程序的结构设计
    步骤0：初始url与调度：spider()
    步骤1：获取页面：downloader()
    步骤2：网页解析：parse()
    步骤3：信息展示：show()
    步骤4：信息保存：save()
使用方法：
    步骤1：继承RequestsSpider
    步骤2：init中配置可选公共变量：proxies, header, 数据库信息(数据使用数据库保存)
    步骤3：在spider中重载url遍历逻辑
    步骤4：在parse中重载页面解析函数
    步骤5：在show中重载信息输出格式
    步骤6：在save中重载信息保存方法
"""

import os
import pickle
import re
import time

import MySQLdb
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from selenium import webdriver

from data.config import DIR_dict
from lib.common import Common
from lib.yeyuc_downloader import YouGet
from lib.yeyuc_multicore import MultiCore


class RequestsSpider():
    def __init__(self, **kwargs):
        ua = UserAgent()
        self.headers = {'User-Agent': ua.random}
        self.proxies = {}

        if kwargs.get('selenium'):
            self.browser = self.selenium()
        pass

    def selenium(self):
        executable_path = os.path.join(DIR_dict.get('EXE_DIR'), 'chromedriver.exe')
        chrome_options = webdriver.ChromeOptions()
        """后台运行Chromedriver"""
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        # chrome_options.add_argument('--start-maximized')
        browser = webdriver.Chrome(executable_path=executable_path, chrome_options=chrome_options)

        """全屏显示"""
        browser.maximize_window()
        time.sleep(5)
        return browser

    def downloader1(self, url):
        self.browser.get(url)
        self.browser.implicitly_wait(20)
        time.sleep(5)
        return self.browser.page_source

    def downloader(self, url):
        try:
            if self.proxies:
                r = requests.get(url, headers=self.headers, proxies=self.proxies)
            else:
                r = requests.get(url, headers=self.headers)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            return r.text
        except:
            return ""

    def parse(self, info, html, **kwargs):
        """
        网页解析: BeautifulSoup + re
        """
        soup = BeautifulSoup(html, "html.parser")
        for li in soup.find_all("li", {"class": "gl-item"}):
            price = eval(
                li.find("div", {
                    "class": "p-price"
                }).text.split("￥")[1])
            name = li.find("div", {
                "class": "p-name p-name-type-2"
            }).find("em").text
            info.append([price, name])

    def show(self, info, **kwargs):
        """
        信息的保存于展示
        """
        tplt = '{:4}\t{:8}\t{:16}'
        print(tplt.format("序号", "价格", "商品名称"))
        count = 0
        for g in info:
            count += 1
            print(tplt.format(count, g[0], g[1]))

    def save(self, info, **kwargs):
        save_name = kwargs.get("save_name", "untitle.pickle")
        path = kwargs.get("save_path", DIR_dict.get("PICKLE_DIR"))
        try:
            with open(path + '\\' + '{0}.pickle'.format(save_name),
                      'wb') as file:
                pickle.dump(info, file)
        except Exception as e:
            print(e)

    def spider(self, **kwargs):
        """
        初始url与调度
        """
        info = []
        goods = "书包"
        start_url = "https://search.jd.com/Search?keyword={0}&enc=utf-8".format(
            goods)

        depth = 2
        for i in range(depth):
            try:
                url = start_url + "&page=" + str(2 * i + 1)
                html = self.downloader(url)
                self.parse(info, html)
            except Exception as e:
                print(e)
        self.show(info)
        self.save(info)


class NewSpider(RequestsSpider):
    def __init__(self, **kwargs):
        RequestsSpider.__init__(self)

        # 配置MySQL数据库
        db_name = 'ip_pool'
        self.mysql_table_name = 'effective'
        self.conn = MySQLdb.connect(host="192.168.1.110",
                                    user="root",
                                    passwd="123",
                                    db=db_name,
                                    charset="utf8")
        self.cursor = self.conn.cursor()

        # 其他变量
        self.start_depth = 0
        self.end_depth = 200
        pass

    def spider(self, **kwargs):
        """
        初始url与调度
        """
        info = []
        effective = []
        # effective = ["http://123.160.1.24:9999", "http://123.160.1.61:9999"]
        start_url = "https://www.xicidaili.com/nn/"
        start_depth = kwargs.get("start_depth", self.start_depth)
        end_depth = kwargs.get("end_depth", self.end_depth)

        for i in range(start_depth, end_depth):
            print("正在爬取第{0}页".format(i + 1))
            try:
                url = start_url + str(i + 1)
                html = self.downloader(url)
                self.parse(info, html)
            except Exception as e:
                print(e)
            self.judge(info, effective)
            self.save(effective)
            self.save2(info)
            info, effective = [], []

    def parse(self, info, html, **kwargs):
        """
        网页解析: BeautifulSoup + re
        """
        soup = BeautifulSoup(html, "html.parser")
        for tr in soup.find("table", {"id": "ip_list"}).find_all("tr")[1:]:
            speed = float(
                tr.find("div", {"class": "bar"})["title"].split("秒")[0])
            if speed > 1:
                continue
            data = re.findall(r"(.+)\n", tr.text)
            ip = data[0]
            port = data[1]
            proxy_type = data[4]
            info.append([ip, port, proxy_type, speed])

    def judge(self, info, effective, **kwargs):
        def is_effective(addr):
            proxies = {
                "http": addr,
                "https": addr,
            }
            try:
                # url = "http://icanhazip.com"
                url = "http://httpbin.org/get"
                r = requests.get(url,
                                 headers=self.headers,
                                 proxies=proxies,
                                 timeout=5)
                r.raise_for_status
                # print(r.text)
                return True
            except Exception as e:
                return False

        for ip_info in info:
            addr = "{0}://{1}:{2}".format(ip_info[2].lower(), ip_info[0],
                                          ip_info[1])
            if is_effective(addr):
                effective.append(addr)
                print(addr)

    def save(self, info, **kwargs):
        try:
            for ip_info in info:
                self.cursor.execute(
                    "insert ignore into {0}(proxy) VALUES('{1}')".format(
                        self.mysql_table_name, ip_info))
                self.conn.commit()
        except Exception as e:
            print(e)


# 给定哔哩哔哩up主投稿页链接，爬取所有视频链接
class NewSpider1(RequestsSpider):
    def __init__(self, **kwargs):
        RequestsSpider.__init__(self, selenium=True)
        self.depth = 1
        # self.depth = 4
        pass

    def spider(self, **kwargs):
        """
        初始url与调度
        """
        info = []
        start_url = "https://space.bilibili.com/434946588/video?tid=0&page="

        for i in range(self.depth):
            print("正在爬取第{0}页".format(i + 1))
            try:
                url = start_url + str(i + 1)
                html = self.downloader1(url)
                self.parse(info, html)
            except Exception as e:
                print(e)
        return info

    def parse(self, info, html, **kwargs):
        """
        网页解析: BeautifulSoup + re
        """
        soup = BeautifulSoup(html, "html.parser")

        nodes = soup.find("ul", {"class": "clearfix cube-list"}).find_all("li")
        aids = [node.get("data-aid") for node in nodes]
        info.extend(aids)


# 给定哔哩哔哩up主投稿页链接，爬取所有视频链接
class NewSpider4(RequestsSpider):
    def __init__(self, **kwargs):
        RequestsSpider.__init__(self, selenium=True)
        self.depth = 1
        # self.depth = 4
        pass

    def spider(self, **kwargs):
        """
        初始url与调度
        """
        info = []
        url = "https://space.bilibili.com/31964921/channel/detail?cid=10974"
        try:
            html = self.downloader1(url)
            self.parse(info, html)
        except Exception as e:
            print(e)
        return info

    def parse(self, info, html, **kwargs):
        """
        网页解析: BeautifulSoup + re
        """
        soup = BeautifulSoup(html, "html.parser")

        nodes = soup.find("ul", {"class": "row video-list clearfix"}).find_all("li")
        aids = [node.get("data-aid") for node in nodes]
        info.extend(aids)


# 给定哔哩哔哩助眠音乐，爬取所有视频链接
class NewSpider2(RequestsSpider):
    def __init__(self, **kwargs):
        RequestsSpider.__init__(self, selenium=True)
        self.depth = 4
        pass

    def spider(self, **kwargs):
        """
        初始url与调度
        """
        info = []
        start_url = "https://search.bilibili.com/all?keyword=%E5%8A%A9%E7%9C%A0&order=click&duration=0&tids_1=0&page="

        for i in range(self.depth):
            print("正在爬取第{0}页".format(i + 1))
            try:
                url = start_url + str(i + 1)
                html = self.downloader1(url)
                self.parse(info, html)
            except Exception as e:
                print(e)
        return info

    def parse(self, info, html, **kwargs):
        """
        网页解析: BeautifulSoup + re
        """

        import re
        soup = BeautifulSoup(html, "html.parser")

        nodes = soup.find("ul", {"class": "video-list clearfix"}).find_all("li")
        aids = [re.findall('av(\d+)?', node.find("a").get("href"))[0] for node in nodes]
        info.extend(aids)


# 爬取佛系资源里的火影忍者的迅雷链接，调用迅雷进行批量下载
class NewSpider3(RequestsSpider):
    def __init__(self, **kwargs):
        RequestsSpider.__init__(self, selenium=True)
        self.depth = 4
        pass

    def spider(self, **kwargs):
        """
        初始url与调度
        """
        info = {}
        url = "http://www.foxiys.com/d-2p14-ftp.html"

        try:
            html = self.downloader1(url)
            self.parse(info, html)
        except Exception as e:
            print(e)
        return info

    def parse(self, info, html, **kwargs):
        """
        网页解析: BeautifulSoup + re
        """

        import re
        soup = BeautifulSoup(html, "html.parser")

        nodes = soup.find("ul", {"class": "row body"}).find_all("li")

        for node in nodes:
            name = re.findall("第(\d+)集", node.text)
            link = re.findall('thunderhref="(thunder://.+?=*)"', str(node))
            if name and link:
                name = int(name[0])
                if name not in info:
                    info[name] = link[0]
        return


class Work(MultiCore):
    def __init__(self, **kwargs):
        self.path = kwargs.get('path')

    def job(self, urls):
        client = YouGet(path=self.path if self.path else DIR_dict.get('RB_DIR'))
        for url in urls:
            client.download(url)


if __name__ == "__main__":
    # common = Common()
    # client = NewSpider3()
    # info = common.order_dict(client.spider(), index=0, reverse=False)
    # client.browser.close()
    #
    #
    # client = Thunder()
    # for item in info.items():
    #     client.download(item[1], item[0])

    path = r'E:\共享文件夹\后端之路\框架学习\Django\繁华嗅'
    # path = r'C:\Users\yeyuc\Desktop\帽子哥'
    # path = r'C:\Users\yeyuc\Desktop\助眠'
    #
    # client = NewSpider1()
    # client = NewSpider2()
    client = NewSpider4()
    urls = client.spider()
    client.browser.close()

    urls = ["https://www.bilibili.com/video/av" + item for item in urls]

    client = Work(path=path)
    data = Common().split_list(urls, 8)
    client.process(data)
    pass
