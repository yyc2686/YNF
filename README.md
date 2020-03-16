## 关于YNF的简单说明

## 项目文件结构：

* bin：启动入口，可执行文件
* docs：存放文档（结果，中间结果，静态文件，配置文件）
* lib：自己编写的第三方库使用方法
* log：日志（项目日志，其他日志）
* tests：测试文件（后期开发）
* README.md：项目描述
* requirements.txt：需要安装的第三方库及版本
* setup.py：管理代码的打包、安装、部署问题。生成虚拟环境"patent"，并安装需要的第三方库。

## data模块说明：

* config：项目各个模块的路径申明，数据库等常用配置
* csv等14个模块，用于存储对应类型的文件
* backup：用于存储数据库备份文件的模块
* recycle_bin：用于存储无用文件或未知文件类型的模块

## lib模块说明：

* common：整理python的常用但不形成规模的函数，封装成类Common
* yeyuc_downloader：集成you_get第三方库、Thunder(迅雷)、multiprocessing(多进程)进行文件下载
* yeyuc_kares：展示使用kares模块进行简单的机器学习模式
* yeyuc_logging：将python日志功能的常用配置封装成一个类LoggingPython
* yeyuc_matplotlib：基于Matplotlib.pyplot，打包各类图形成类
* yeyuc_mongo：将pymongo的常用配置封装成一个类MongoPython
* yeyuc_multicore：将multiprocessing的常用配置封装成一个类MultiCore
* yeyuc_mysql：将MySQLdb的常用配置封装成一个类MysqlPython
* yeyuc_networkx：整理networkx的常用库函数，实现复杂网络中的一些指标，并封装成类NetworkxPython
* yeyuc_read_write：整理python常用的文件类型的读写函数，并封装成类ReadWrite
* yeyuc_sklearn：展示使用sklearn模块进行机器学习的模式
* yeyuc_spider：整理python爬虫的基本过程，封装成类