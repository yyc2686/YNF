#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/28 20:20
# @Author  : yeyuc
# @Email   : yeyucheng_uestc@163.com
# @File    : main.py
# @Software: PyCharm
"""
    __project_ = main
    __file_name__ = main
    __author__ = yeyuc
    __time__ = 2019/12/28 20:20
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
import os
import time
from functools import wraps

import PIL.Image as image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn import datasets
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA, NMF
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve, validation_curve, train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data.config import DIR_dict
from lib.common import Common


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


class Sklearn(Common):
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
                    # imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
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
            X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)

        elif kwargs.get("make_classification"):
            X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2, random_state=22,
                                       n_clusters_per_class=1, scale=100)

        ###绘制构造的数据###
        if kwargs.get("show"):
            if kwargs.get("make_regression"):
                self.axes.scatter(X, y)
            elif kwargs.get("make_classification"):
                self.axes.scatter(X[:, 0], X[:, 1], c=y)
            self.fig_show()
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
        if kwargs.get("KNN"):
            model = KNeighborsClassifier()
        elif kwargs.get("DT"):
            model = DecisionTreeClassifier()
        elif kwargs.get("GNB"):
            model = GaussianNB()
        elif kwargs.get("MLP"):
            model = MLPClassifier(hidden_layer_sizes=(kwargs.get("hidden_layer_sizes", 100),),
                                  activation=kwargs.get("activation", 'logistic'),
                                  solver=kwargs.get("solver", 'adam'),
                                  learning_rate_init=kwargs.get("learning_rate_init", 0.0001),
                                  max_iter=kwargs.get("max_iter", 2000))

        # 回归
        elif kwargs.get("SVC"):
            model = SVC(gamma='auto', kernel=kwargs.get("kernel"))
        elif kwargs.get("LinearRegression"):
            model = LinearRegression()
        elif kwargs.get("Ridge"):
            model = Ridge(alpha=kwargs.get("alpha", 1.0), fit_intercept=kwargs.get("fit_intercept", True))

        # 聚类
        elif kwargs.get("KMeans"):
            model = KMeans(n_clusters=kwargs.get("n_clusters", 4))
        elif kwargs.get("DBSCAN"):
            model = DBSCAN(eps=kwargs.get("eps", 0.01), min_samples=kwargs.get("min_samples", 20))

        # 降维
        elif kwargs.get("PCA"):
            model = PCA(n_components=kwargs.get("n_components", 2), whiten=kwargs.get("whiten", False))
        elif kwargs.get("NMF"):
            model = NMF(n_components=kwargs.get("n_components", 2), init=kwargs.get("init", 'nndsvda'),
                        tol=kwargs.get("tol", 5e-3))

        model.fit(kwargs.get("X_train"), kwargs.get("y_train").ravel())
        return model

    ###预测###
    def predict(self, model, **kwargs):
        return model.predict(kwargs.get("X"))

    ###打分###
    def score(self, model, **kwargs):
        return model.score(kwargs.get("X"), kwargs.get("y"))

    ###绘图###
    def fig_show(self, **kwargs):
        self.fig.show()
        if not kwargs.get("move_on"):
            self.fig, self.axes = plt.subplots()

    ###交叉验证###
    def cross_valid(self, **kwargs):
        ###训练数据###
        X = kwargs.get("X")
        y = kwargs.get("y")

        # 引入交叉验证,数据分为5组进行训练
        if kwargs.get("common"):
            knn = KNeighborsClassifier(n_neighbors=5)  # 选择邻近的5个点
            scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')  # 评分方式为accuracy
            print(scores)  # 每组的评分结果
            print(scores.mean())  # 平均评分结果

        ###设置n_neighbors的值为1到30,通过绘图来看训练分数###
        if kwargs.get("valid_neighbors"):
            k_range = range(1, 31)
            k_score = []
            for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k)
                scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')  # for classfication
                k_score.append(scores.mean())
            self.axes.plot(k_range, k_score)
            self.axes.set_xlabel('Value of k for KNN')
            self.axes.set_ylabel('CrossValidation accuracy')
            self.fig_show()

        elif kwargs.get("neg_mean_squared_error"):
            k_range = range(1, 31)
            k_score = []
            for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k)
                loss = -cross_val_score(knn, X, y, cv=10, scoring='neg_mean_squared_error')  # for regression
                k_score.append(loss.mean())
            self.axes.plot(k_range, k_score)
            self.axes.set_xlabel('Value of k for KNN')
            self.axes.set_ylabel('neg_mean_squared_error')
            self.fig_show()

    ###过拟合处理###
    def over_fitting(self, **kwargs):
        # 引入数据
        X = self.digits.data
        y = self.digits.target

        if kwargs.get("common"):
            # train_size表示记录学习过程中的某一步,比如在10%,25%...的过程中记录一下
            train_size, train_loss, test_loss = learning_curve(
                SVC(gamma=0.1), X, y, cv=10, scoring='neg_mean_squared_error',
                train_sizes=[0.1, 0.25, 0.5, 0.75, 1]
            )
        elif kwargs.get("valid_gamma"):
            # 改变param来观察Loss函数情况
            param_range = np.logspace(-6, -2.3, 5)
            train_loss, test_loss = validation_curve(
                SVC(), X, y, param_name='gamma', param_range=param_range, cv=10,
                scoring='neg_mean_squared_error'
            )

        train_loss_mean = -np.mean(train_loss, axis=1)
        test_loss_mean = -np.mean(test_loss, axis=1)

        if kwargs.get("common"):
            # 将每一步进行打印出来
            self.axes.plot(train_size, train_loss_mean, 'o-', color='r', label='Training')
            self.axes.plot(train_size, test_loss_mean, 'o-', color='g', label='Cross-validation')
        elif kwargs.get("valid_gamma"):
            self.axes.plot(param_range, train_loss_mean, 'o-', color='r', label='Training')
            self.axes.plot(param_range, test_loss_mean, 'o-', color='g', label='Cross-validation')
            self.axes.set_xlabel('gamma')
            self.axes.set_ylabel('loss')
        self.axes.legend(loc='best')
        self.fig_show()

    #####实例#####
    # 分类
    def iris_classification(self):
        ###引入数据###
        # iris_X = self.iris.data  # 特征变量
        # iris_y = self.iris.target  # 目标变量
        X, y = self.import_data(iris=True)

        # 利用train_test_split进行将训练集和测试集进行分开，test_size占30%
        X_train, X_test, y_train, y_test = self.preprocess(X=X, y=y, train_test_split=True)

        ###训练数据###
        model = self.create_model(KNN=True, X_train=X_train, y_train=y_train)

        ###预测数据###
        print(model.predict(X_test))
        print(y_test)
        print(model.score(X, y))
        pass

    def posture_classification(self):
        # 导入数据
        X_train, X_test, y_train, y_test = self.import_private_data(posture=True)
        # 使用全部数据作为训练集，借助train_test_split将训练数据打乱
        # X_train, a, y_train, b = train_test_split(X_train, y_train, test_size=1e-20)

        # 建立模型
        model_knn = self.create_model(X_train=X_train, y_train=y_train, KNN=True)
        model_dt = self.create_model(X_train=X_train, y_train=y_train, DT=True)
        model_gnb = self.create_model(X_train=X_train, y_train=y_train, GNB=True)

        answer_dt = model_dt.predict(X_test)
        answer_knn = model_knn.predict(X_test)
        answer_gnb = model_gnb.predict(X_test)

        ###############################################################################
        # 结果展示：计算准确率与召回率
        print("\n\nThe classification report for KNN:")
        print(classification_report(y_test, answer_knn))
        print("\n\nThe classification report for DT:")
        print(classification_report(y_test, answer_dt))
        print("\n\nThe classification report for GNB:")
        print(classification_report(y_test, answer_gnb))

    def stock_predict(self):
        # 数据导入
        X, y = self.import_private_data(stock=True)

        # 建立模型
        # 调用svm函数,并设置kernel参数,默认是rbf,其它:'linear','poly','sigmoid'
        model = self.create_model(X_train=X, y_train=y, SVC=True, kernel='rbf')
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')  # 评分方式为accuracy
        print(scores)  # 每组的评分结果
        print(scores.mean())  # 平均评分结果

    # 回归
    def boston_regression(self):
        ###引入数据###
        X, y = self.import_data(boston=True)
        X_train, X_test, y_train, y_test = self.preprocess(X=X, y=y, train_test_split=True)

        ###训练数据###
        model = self.create_model(LinearRegression=True, X_train=X_train, y_train=y_train)

        print(y[:4])  # 预测前4个数据
        # print(model.predict(X[:4, :]))  # 预测前4个数据
        print(self.predict(model=model, X=X[:4, :]))  # 预测前4个数据

        ###属性和功能###
        # (506, 13)data_X共13个特征变量
        # print(data_X.shape)
        # print(model.coef_)
        # print(model.intercept_)
        # print(model.get_params())  # 得到模型的参数
        # print(model.score(X, y))  # 对训练情况进行打分
        print(self.score(model=model, X=X, y=y))  # 对训练情况进行打分
        pass

    def house_price_regression(self, **kwargs):
        # 导入数据
        X, y = self.import_private_data(house_price=True)
        X_plot = np.arange(min(X), max(X)).reshape([-1, 1])

        # 多项式回归
        if kwargs.get("poly"):
            poly_reg = PolynomialFeatures(degree=kwargs.get("degree", 2))
            X_poly = poly_reg.fit_transform(X)

        # 建立模型
        model = self.create_model(X_train=X_poly, y_train=y, LinearRegression=True)

        # 结果展示
        self.axes.scatter(X, y, color='red')
        self.axes.plot(X_plot, model.predict(poly_reg.fit_transform(X_plot)), color='blue')
        self.axes.set_xlabel('Area')
        self.axes.set_ylabel('Price')
        self.fig.show()

    def traffic_regression(self, **kwargs):
        # 导入数据
        X, y = self.import_private_data(traffic=True)

        if kwargs.get("poly"):
            poly_reg = PolynomialFeatures(degree=kwargs.get("degree", 6))
            X_poly = poly_reg.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=0)

        # 建立模型
        model = self.create_model(X_train=X_train, y_train=y_train, Ridge=True)

        # 结果展示
        print(model.score(X_test, y_test))

        start = 200
        end = 300
        y_pre = model.predict(X_poly)
        time = np.arange(start, end)
        self.axes.plot(time, y[start:end], 'b', label='real')
        self.axes.plot(time, y_pre[start:end], 'r', label='predict')
        self.axes.legend(loc='upper left')
        self.fig.show()

    def preprocess_works(self, **kwargs):
        ###生成的数据如下图所示###
        X, y = self.make_data(make_classification=True)

        ###利用minmax方式对数据进行规范化###
        if kwargs.get("scale"):
            X = self.preprocess(X, MinMaxScaler=True)  # feature_range=(-1,1)可设置重置范围

        X_train, X_test, y_train, y_test = self.preprocess(X=X, y=y, train_test_split=True)
        model = self.create_model(X_train=X_train, y_train=y_train, SVC=True)
        print(model.score(X_test, y_test))
        print(self.score(model=model, X=X_test, y=y_test))

    def handwriting_recognition(self, **kwargs):
        # 导入数据
        X_train, X_test, y_train, y_test = self.import_private_data(handwriting=True)

        # 建立模型
        if kwargs.get("KNN"):
            model = self.create_model(X_train=X_train, y_train=y_train, KNN=True)
        elif kwargs.get("MLP"):
            model = self.create_model(X_train=X_train, y_train=y_train, MLP=True)
        # 结果展示
        res = model.predict(X_test)  # 对测试集进行预测
        error_num = np.sum(res != y_test)  # 统计分类错误的数目
        num = len(X_test)  # 测试集的数目
        print("Total num:", num, " Wrong num:", \
              error_num, "  WrongRate:", error_num / float(num))

    # 聚类
    def image_segmentation(self, name="starbucks.jpg", save_name="result-bull-starbucks.jpg"):
        # 导入图片数据
        path = os.path.join(DIR_dict.get("PICTURE_DIR"), name)
        imgData, row, col = self.import_private_data(image_segmentation=True, path=path)

        # 创建模型
        model = self.create_model(X_train=imgData, KMeans=True, n_clusters=4)
        label = model.fit_predict(imgData)

        # 数据展示
        label = label.reshape([row, col])
        pic_new = image.new("L", (row, col))
        for i in range(row):
            for j in range(col):
                pic_new.putpixel((i, j), int(256 / (label[i][j] + 1)))
        pic_new.save(os.path.join(DIR_dict.get("PICTURE_DIR"), save_name, "JPEG"))

    def province_classification(self, **kwargs):
        # 数据准备
        n_clusters = kwargs.get("n_clusters", 2)
        X, y = self.import_private_data(province_expense=True)

        # 建立模型，完成分类
        model = self.create_model(X_train=X, KMeans=True, n_clusters=n_clusters)
        label = model.predict(X)

        # 结果展示
        expenses = np.sum(model.cluster_centers_, axis=1)
        CityCluster = [[] for i in range(n_clusters)]
        for i in range(len(y)):
            CityCluster[label[i]].append(y[i])
        data = {expenses[i]: CityCluster[i] for i in range(n_clusters)}
        data = self.order_dict_by_key(dict=data, reverse=True)

        print(list(data))
        for expense in data:
            print("Expenses:%.2f" % expense)
            print(data.get(expense))
        return True

    def online_mode_classification(self, **kwargs):
        # 数据准备
        # n_clusters = kwargs.get("n_clusters", 2)
        X = self.import_private_data(online_times=True)

        X = X[:, 0:1]
        # 建立模型，完成分类
        model = self.create_model(X_train=X, DBSCAN=True, eps=0.01, min_samples=20)
        label = model.labels_

        # 结果展示
        print('Labels:')
        print(label)
        raito = len(label[label[:] == -1]) / len(label)
        print('Noise raito:', format(raito, '.2%'))

        n_clusters_ = len(set(label)) - (1 if -1 in label else 0)

        print('Estimated number of clusters: %d' % n_clusters_)
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, label))

        for i in range(n_clusters_):
            print('Cluster ', i, ':')
            print(list(X[label == i].flatten()))

        self.axes.hist(X, 24)
        self.fig.show()

        return True

    # 降维
    def iris_PCA(self):
        # 导入数据
        X, y = self.import_data(iris=True)

        # 创建模型
        model = self.create_model(X_train=X, PCA=True)
        reduced_X = model.fit_transform(X)

        # 结果展示
        red_x, red_y = [], []
        blue_x, blue_y = [], []
        green_x, green_y = [], []

        for i in range(len(reduced_X)):
            if y[i] == 0:
                red_x.append(reduced_X[i][0])
                red_y.append(reduced_X[i][1])
            elif y[i] == 1:
                blue_x.append(reduced_X[i][0])
                blue_y.append(reduced_X[i][1])
            else:
                green_x.append(reduced_X[i][0])
                green_y.append(reduced_X[i][1])

        self.axes.scatter(red_x, red_y, c='r', marker='x')
        self.axes.scatter(blue_x, blue_y, c='b', marker='D')
        self.axes.scatter(green_x, green_y, c='g', marker='.')
        self.fig.show()

    def faces_NMF(self):
        n_row, n_col = 2, 3
        n_components = n_row * n_col
        image_shape = (64, 64)

        ###############################################################################
        # 导入数据
        faces = self.import_data(LabeledFaces=True)

        # 建立模型
        model_PCA = self.create_model(X_train=faces, PCA=True, n_components=6, whiten=True)
        components_PCA = model_PCA.components_

        model_NMF = self.create_model(X_train=faces, NMF=True, n_components=6, init='nndsvda', tol=5e-3)
        components_NMF = model_NMF.components_

        ###############################################################################
        # 结果展示--Target Face
        def plot_gallery(title, images, n_col=n_col, n_row=n_row):
            plt.figure(figsize=(2. * n_col, 2.26 * n_row))
            plt.suptitle(title, size=16)

            for i, comp in enumerate(images):
                plt.subplot(n_row, n_col, i + 1)
                vmax = max(comp.max(), -comp.min())

                plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                           interpolation='nearest', vmin=-vmax, vmax=vmax)
                plt.xticks(())
                plt.yticks(())
            plt.subplots_adjust(0.01, 0.05, 0.99, 0.94, 0.04, 0.)

        plot_gallery("First centered Olivetti faces", faces[:n_components])
        plot_gallery('Eigenfaces - PCA using randomized SVD', components_PCA[:n_components])
        plot_gallery('Non-negative components - NMF', components_NMF[:n_components])
        plt.show()

        # ###############################################################################
        # # 比较两种模型，参数配置如下：
        # estimators = [
        #     ('Eigenfaces - PCA using randomized SVD',
        #      PCA(n_components=6, whiten=True)),
        #
        #     ('Non-negative components - NMF',
        #      NMF(n_components=6, init='nndsvda', tol=5e-3))
        # ]
        #
        # ###############################################################################
        #
        # for name, estimator in estimators:
        #     print("Extracting the top %d %s..." % (n_components, name))
        #     print(faces.shape)
        #     estimator.fit(faces)
        #     components_ = estimator.components_
        #     plot_gallery(name, components_[:n_components])
        #
        # plt.show()


@fn_timer
def main():
    client = Sklearn()
    client.boston_regression()
    # client.house_price_regression()
    # client.traffic_regression()
    # client.preprocess_works()
    # client.handwriting_recognition()

    # client.handwriting_recognition(MLP=True)
    # client.handwriting_recognition(KNN=True)
    # client.traffic_regression(poly=True, degree=6)
    # client.house_price_regression(poly=True, degree=2)
    # client.stock_predict()
    # client.posture_classification()
    # client.image_segmentation()
    # client.faces_NMF()
    # client.iris_PCA()
    # client.online_mode_classification()
    # client.province_classification(n_clusters=4)
    # client.iris_classification()
    # client.linear_regression()
    # client.preprocess_works()
    # client.control()

    pass


if __name__ == '__main__':
    main()
