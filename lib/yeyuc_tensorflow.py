#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/1/7 21:34
# @Author  : yeyuc
# @Email   : yeyucheng_uestc@163.com
# @File    : main.py
# @Software: PyCharm
"""
    __project_ = main
    __file_name__ = main
    __author__ = yeyuc
    __time__ = 2020/1/7 21:34
    Code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃        ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
import time
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import datasets
from sklearn.datasets.samples_generator import make_classification

from utils.public_function import PublicFunction


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        cost_time = t1 - t0
        if cost_time <= 1:
            print("本次运行耗时：{0}秒 ".format(str(cost_time)))
        elif cost_time <= 60:
            print("本次运行耗时：{0}秒 ".format(str(cost_time)))
        else:
            print('本次运行耗时：{0}时{1}分{2}秒 '.format(int(cost_time / 3600), int((cost_time / 60) % 60), int(cost_time % 60)))
        return result

    return function_timer


class TensorFlow(PublicFunction):
    def __init__(self):
        self.fig, self.axes = plt.subplots()

    #####Sklearn通用学习模式#####
    def control(self, **kwargs):
        if kwargs.get("make_data"):
            ###构造数据###
            X, y = self.make_data()
        else:
            ###引入数据###
            X, y = self.import_data()

        if not kwargs.get("preprocess"):
            ###数据预处理###
            X = self.preprocess(array=X, MinMaxScaler=True)
        X_train, X_test, y_train, y_test = self.preprocess(X=X, y=y, train_test_split=True)

        ###读取模型###
        if kwargs.get("import_model"):
            model = self.model_read(save_name=kwargs.get("save_name"),
                                    save_path=kwargs.get("save_path"))
        else:
            ###训练模型###
            model = self.create_model(X_train=X_train, y_train=y_train)

        ###预测###
        y_predict = self.predict(model=model, X=X_test)

        ###打分###
        if kwargs.get("score"):
            score = self.score(model=model, X=X_test, y=y_test)

        ###绘图###
        if kwargs.get("fig_show"):
            self.fig_show()

        ###交叉验证###
        if kwargs.get("cross_valid"):
            self.cross_valid()

        ###过拟合处理###
        if kwargs.get("over_fitting"):
            self.over_fitting()

        ###保存模型###
        if kwargs.get("model_save"):
            self.model_save(model=model, save_name=kwargs.get("save_name"),
                            save_path=kwargs.get("save_path"))
        return True

    ###引入数据###
    def import_private_data(self, **kwargs):
        if kwargs.get("province_expense"):
            def loadData(filePath):
                fr = open(filePath, 'r+', encoding='gbk')
                lines = fr.readlines()
                retData = []
                retCityName = []
                for line in lines:
                    items = line.strip().split(",")
                    retCityName.append(items[0])
                    retData.append([float(items[i]) for i in range(1, len(items))])
                return retData, retCityName

            path = os.path.join(DIR_dict.get("TXT_DIR"), "city.txt")
            return loadData(path)

        elif kwargs.get("online_times"):
            def loadData(filePath):
                mac2id = dict()
                onlinetimes = []
                f = open(filePath, encoding='utf-8')
                for line in f:
                    mac = line.split(',')[2]
                    onlinetime = int(line.split(',')[6])
                    starttime = int(line.split(',')[4].split(' ')[1].split(':')[0])
                    if mac not in mac2id:
                        mac2id[mac] = len(onlinetimes)
                        onlinetimes.append((starttime, onlinetime))
                    else:
                        onlinetimes[mac2id[mac]] = [(starttime, onlinetime)]
                real_X = np.array(onlinetimes).reshape((-1, 2))
                return real_X

            path = os.path.join(DIR_dict.get("TXT_DIR"), "online_times.txt")
            return loadData(path)

        elif kwargs.get("image_segmentation"):
            def loadData(filePath):
                f = open(filePath, 'rb')
                data = []
                img = image.open(f)
                m, n = img.size
                for i in range(m):
                    for j in range(n):
                        x, y, z = img.getpixel((i, j))[:3]
                        data.append([x / 256.0, y / 256.0, z / 256.0])
                f.close()
                return np.mat(data), m, n

            return loadData(filePath=kwargs.get("path"))

        elif kwargs.get("posture"):
            def load_dataset(feature_paths, label_paths):
                """
                读取特征文件列表和标签文件列表中的内容，归并后返回
                :param feature_paths:
                :param label_paths:
                :return:
                """
                # 定义空的标签变量label，特征数组feature
                feature = np.ndarray(shape=(0, 41))
                label = np.ndarray(shape=(0, 1))

                for file in feature_paths:
                    # 使用逗号分隔符读取特征数据，将问号替换标记为缺失值，文件中不包含表头
                    df = pd.read_table(file, delimiter=',', na_values='?', header=None)
                    # 使用平均值补全缺失值，然后将数据进行补全
                    imp = SimpleImputer(missing_values="NaN", strategy="mean")
                    imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
                    imp.fit(df)
                    df = imp.transform(df)
                    # 将新读入的数据合并到特征集合中
                    feature = np.concatenate((feature, df))

                for file in label_paths:
                    # 读入标签数据，文件中不包含表头
                    df = pd.read_table(file, header=None)
                    # 将新读入的数据合并到标签集合中
                    label = np.concatenate((label, df))

                return feature, label

            # 设置数据路径
            feature_paths = []
            label_paths = []
            paths = [os.path.join(DIR_dict.get("TXT_DIR"), "posture") + '\\' + letter for letter in
                     ["A", "B", "C", "D", "E"]]
            for path in paths:
                for file in os.listdir(path):
                    if ".feature" in file:
                        feature_paths.append(os.path.join(path, file))
                    elif ".label" in file:
                        label_paths.append(os.path.join(path, file))
            # 将前4个数据作为训练集读入
            X_train, y_train = load_dataset(feature_paths=feature_paths[:4], label_paths=label_paths[:4])
            # 将最后一个数据作为测试集读入
            X_test, y_test = load_dataset(feature_paths=feature_paths[4:], label_paths=label_paths[4:])
            return X_train, X_test, y_train, y_test

        elif kwargs.get("stock"):
            # read_csv:参数一:数据源.encoding:编码格式.parse_dates:第n列解析为日期.index_col:用作索引的列编号
            # sort_index:参数一:按0列排,ascending(true)升序,inplace:排序后是否覆盖原数据
            data = pd.read_csv(os.path.join(DIR_dict.get("CSV_DIR"), '000777.csv'), encoding='gbk', parse_dates=[0],
                               index_col=0)
            data.sort_index(0, ascending=True, inplace=True)

            # dayfeature:选取150天的数据
            # featurenum:选取5个特征*天数
            # x:记录150天的5个特征值
            # y:记录涨或者跌
            # data.shape[0]-dayfeature:因为我们要用150天数据做训练,对于条目为200条的数据,只有50条数据有前150天的数据来训练的,所以训练集的大小就是200-150
            # 对于每一条数据,他的特征是前150天的甩有特征数据,即150*5,+1是将当天的开盘价引入作为一条特征数据
            dayfeature = 150
            featurenum = 5 * dayfeature
            x = np.zeros((data.shape[0] - dayfeature, featurenum + 1))
            y = np.zeros((data.shape[0] - dayfeature))

            for i in range(0, data.shape[0] - dayfeature):
                x[i, 0:featurenum] = np.array(data[i:i + dayfeature] \
                                                  [['收盘价', '最高价', '最低价', '开盘价', '成交量']]).reshape((1, featurenum))
                x[i, featurenum] = data.iloc[i + dayfeature]['开盘价']

            for i in range(0, data.shape[0] - dayfeature):
                if data.iloc[i + dayfeature]['收盘价'] >= data.iloc[i + dayfeature]['开盘价']:
                    y[i] = 1
                else:
                    y[i] = 0
            return x, y

        elif kwargs.get("house_price"):
            X = []
            y = []

            with open(os.path.join(DIR_dict.get("TXT_DIR"), 'prices.txt'), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    items = line.strip().split(',')
                    X.append(int(items[0]))
                    y.append(int(items[1]))

            length = len(X)
            X = np.array(X).reshape([length, 1])
            y = np.array(y)
            return X, y

        elif kwargs.get("traffic"):
            data = np.genfromtxt(os.path.join(DIR_dict.get("CSV_DIR"), 'traffic.csv'), delimiter=',', skip_header=True)
            X = data[:, 1:5]
            y = data[:, 5]
            return X, y

        elif kwargs.get("handwriting"):
            def img2vector(fileName):
                retMat = np.zeros([1024], int)  # 定义返回的矩阵，大小为1*1024
                with open(fileName) as file:
                    lines = file.readlines()  # 读取文件的所有行
                    for i in range(32):  # 遍历文件所有行
                        for j in range(32):  # 并将01数字存放在retMat中
                            retMat[i * 32 + j] = lines[i][j]
                return retMat

            def readDataSet(path):
                fileList = os.listdir(path)  # 获取文件夹下的所有文件
                numFiles = len(fileList)  # 统计需要读取的文件的数目
                dataSet = np.zeros([numFiles, 1024], int)  # 用于存放所有的数字文件
                hwLabels = np.zeros([numFiles])  # 用于存放对应的标签(与神经网络的不同)
                for i in range(numFiles):  # 遍历所有的文件
                    filePath = fileList[i]  # 获取文件名称/路径
                    digit = int(filePath.split('_')[0])  # 通过文件名获取标签
                    hwLabels[i] = digit  # 直接存放数字，并非one-hot向量
                    dataSet[i] = img2vector(path + '/' + filePath)  # 读取文件内容
                return dataSet, hwLabels
                # read dataSet

            path = os.path.join(DIR_dict.get("TXT_DIR"), "digits")
            train_dataSet, train_hwLabels = readDataSet(path=os.path.join(path, 'trainingDigits'))
            test_dataSet, test_hwLabels = readDataSet(path=os.path.join(path, 'testDigits'))
            return train_dataSet, test_dataSet, train_hwLabels, test_hwLabels

    def import_data(self, **kwargs):
        if kwargs.get("newsgroups"):
            """5.7：The 20 newsgroups text dataset：18000个新闻、可分为20个topics；已经分为training和testing集。"""
            # newsgroups_train = datasets.fetch_20newsgroups(subset='train')
            cats = ['alt.atheism', 'sci.space']
            newsgroups_train = datasets.fetch_20newsgroups(subset='train', categories=cats)
            return newsgroups_train
        elif kwargs.get("LabeledFaces"):
            """
            5.9：The Labeled Faces in the Wild face recognition dataset：JPEG人脸图片，
                典型任务1（Face Verification）：给出两张图，判断出是否是同一个人；
                典型任务2（Face Recognition）：给出一张图，判断出是哪个人。
            """
            dataset = datasets.fetch_olivetti_faces(shuffle=True, random_state=RandomState(0))
            faces = dataset.data
            return faces
        elif kwargs.get("iris"):
            iris = datasets.load_iris()
            return iris.data, iris.target
        elif kwargs.get("digits"):
            digits = datasets.load_digits()
            return digits.data, digits.target
        elif kwargs.get("boston"):
            boston = datasets.load_boston()
            return boston.data, boston.target

    ###构造数据###
    def make_data(self, **kwargs):
        if kwargs.get("make_regression"):
            X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=1)

        elif kwargs.get("make_classification"):
            X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2, random_state=22,
                                       n_clusters_per_class=1, scale=100)
        elif kwargs.get("quadratic_function"):
            X = np.random.rand(100).astype(np.float32)
            y = X ** 2 + 0.3

        return X, y

    ###数据预处理###
    def preprocess(self, **kwargs):
        """
        :param array: numpy矩阵
        :param kwargs: 归一化的方法
        :return: 归一化之后的矩阵
        """
        array = kwargs.get("array")
        # 零均值单位方差
        if kwargs.get("StandardScaler"):
            return preprocessing.scale(array)

        #  MinMaxScaler(最小最大值标准化)
        elif kwargs.get("MinMaxScaler"):
            return preprocessing.MinMaxScaler().fit_transform(array)

        #  MaxAbsScaler（绝对值最大标准化）
        elif kwargs.get("MaxAbsScaler"):
            return preprocessing.MaxAbsScaler().fit_transform(array)

        #  将数据集分为训练集和测试集
        if kwargs.get("train_test_split"):
            return train_test_split(kwargs.get("X"), kwargs.get("y"),
                                    test_size=kwargs.get("test_size", 0.3))

    ###训练模型###
    def create_model(self, **kwargs):
        # 分类
        # 回归
        if kwargs.get("Sequential"):
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
            model.summary()
            model.compile(
                optimizer=kwargs.get("optimizer", "adam"),
                loss=kwargs.get("loss", "mse")
            )

        # 聚类
        # 降维

        model.fit(kwargs.get("X_train"), kwargs.get("y_train"), epochs=kwargs.get("epochs", 500))
        return model

    ###保存模型###
    def predict(self, model, **kwargs):
        return model.predict(kwargs.get("X"))

    ###预测###
    def predict(self, model, **kwargs):
        return model.predict(kwargs.get("X"))

    #####实例#####
    def linear_predict(self):
        # 引入数据
        X, y = self.make_data(make_regression=True)

        # 建立模型
        model = self.create_model(X_train=X, y_train=y, Sequential=True, optimizer="adam", loss="mse", epochs=50000)

        # 结果展示
        self.axes.scatter(X, y)
        self.axes.scatter(X, model.predict(X))
        self.fig.show()
        pass


@fn_timer
def main():
    client = TensorFlow()
    client.linear_predict()

    pass


if __name__ == '__main__':
    main()
