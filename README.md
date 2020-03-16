## 要点

- [x] 1.软件定位，软件的基本功能
- [x] 2.运行代码方法：安装环境、启动命令等
- [x] 3.简要使用说明
- [x] 4.代码目录结构说明
- [x] 5.常见问题说明

## 项目文件结构：

* bin：启动入口，可执行文件
* docs：存放文档（结果，中间结果，静态文件，配置文件）
* lib：自己编写的第三方库使用方法
* log：日志（项目日志，其他日志）
* tests：测试文件
* README.md：项目描述
* requirements.txt：需要安装的第三方库及版本
* setup.py：管理代码的打包、安装、部署问题。生成虚拟环境"patent"，并安装需要的第三方库。

## 注意

* 本项目基于python3，推荐使用python3.6。版本不一致，安装第三方库可能失败。

  * 转到：[Unofficial Windows Binaries for Python Extension Packages](https://www.lfd.uci.edu/~gohlke/pythonlibs/)，自行下载并安装对应python版本的第三方库whl文件。

  * 打开命令行，

    ```
    cd whl文件路径
    pip install whl文件.whl
    ```

    