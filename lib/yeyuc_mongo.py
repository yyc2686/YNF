# -*- coding: utf-8 -*-

# @Time    : 2020/2/2
# @Author  : yeyuc
# @Email   : yeyucheng_uestc@163.com
# @File    : yeyuc_mongo.py
# @Software: PyCharm

"""
概要：将pymongo的常用配置封装成一个类MongoPython

使用方法：

    步骤1：实例化MongoPython，示例：client = MongoPython(db='test')
    步骤2：调用client的函数，完成增删查改等操作

常用函数：
    client.SetDatabase(db, user='', passwd='')                               设置（修改）当前数据库

    # 集合操作 ---------------------------------------------------------------------------------------------------
    client.GetCollections()                                            获取当前数据库的全部集合
    client.GetCollectionsDict()                                        获取当前数据库的全部集合与文档数量
    client.CreateCollection(collection)                                创建集合
    client.DropCollection(collection)                                  删除集合
    client.ImportCollection(col, file_name)                            导入集合
    client.ExportCollection(col, file_name)                            导出集合

    # 文档操作 ---------------------------------------------------------------------------------------------------
    client.InsertDoc(collection, data)                                 data为字典时，单条插入，data为列表时，批量插入
    client.InsertBatchDoc(collection, data)                            data为字典时，单条插入，data为列表时，批量插入
    client.RemoveDoc(collection, docFilter=None)                       删除文档，docFilter=None时删除集合的全部文档
    client.UpdateDoc(collection, docFilter, data, modifier=None)       更新文档，支持使用$inc/$set/$unset等修改器
    client.UpsertDoc(collection, docFilter, data)                      如果文档不存在，则插入文档；如果文档存在，则更新文档
    client.GetDoc(collection, docFilter=None, colFilter=None)          返回单个文档
    client.CountDoc(collection, docFilter=None)                        返回集合或查询的文档总数
    client.GetCursor(collection, docFilter=None, colFilter=None)       返回多个文档的游标

    # 索引操作 ---------------------------------------------------------------------------------------------------
    client.IndexInformation(collection)                                获取集合的索引信息
    client.EnsureIndex(collection, key_or_list)                        检查索引是否存在，若不存在，则创建索引
    client.CreateIndex(collection, key_or_list)                        创建索引
    client.DropIndex(collection, key=None)                             删除索引，key=None时删除全部索引（_id除外）

全部函数：
    # MongoDB客户端类
    # pymongo是python访问MongoDB的模块，使用该模块，我们定义了一个操作MongoDB的类PyMongoClient，包含了连接管理、集合管理、索引管理、增删改查、文件操作、聚合操作等方法。
    # ---------------------------------------------------------------------------------------------------
    # PyMongoClient(host='localhost', port='27017', db='test', user=None, passwd=None, loop=5, rate=8)
    # ---------------------------------------------------------------------------------------------------
    # PyMongoClient.SetDatabase(db, user, passwd)                               设置（修改）当前数据库
    # PyMongoClient.CloseConnection()                                           关闭连接
    # PyMongoClient.Logout()                                                    注销用户
    # PyMongoClient.GetStatus()                                                 获取MogoDB服务器的状态
    # PyMongoClient.IsMongos()                                                  判断是否是MongoS
    # PyMongoClient.GetDateTime()                                               获取MongoDB服务器的当前时间（需要权限支持）
    # ---------------------------------------------------------------------------------------------------
    # PyMongoClient.GetCollections()                                            获取当前数据库的全部集合
    # PyMongoClient.CreateCollection(collection)                                创建集合
    # PyMongoClient.DropCollection(collection)                                  删除集合
    # ---------------------------------------------------------------------------------------------------
    # PyMongoClient.IndexInformation(collection)                                获取集合的索引信息
    # PyMongoClient.EnsureIndex(collection, key_or_list)                        检查索引是否存在，若不存在，则创建索引
    # PyMongoClient.CreateIndex(collection, key_or_list)                        创建索引
    # PyMongoClient.DropIndex(collection, key=None)                             删除索引，key=None时删除全部索引（_id除外）
    # ---------------------------------------------------------------------------------------------------
    # PyMongoClient.InsertDoc(collection, data)                                 data为字典时，单条插入，data为列表时，批量插入
    # PyMongoClient.RemoveDoc(collection, docFilter=None)                       删除文档，docFilter=None时删除集合的全部文档
    # PyMongoClient.UpdateDoc(collection, docFilter, data, modifier=None)       更新文档，支持使用$inc/$set/$unset等修改器
    # PyMongoClient.UpsertDoc(collection, docFilter, data)                      如果文档不存在，则插入文档；如果文档存在，则更新文档
    # PyMongoClient.GetDoc(collection, docFilter=None, colFilter=None)          返回单个文档
    # PyMongoClient.CountDoc(collection, docFilter=None)                        返回集合或查询的文档总数
    # PyMongoClient.GetCursor(collection, docFilter=None, colFilter=None)       返回多个文档的游标
    # PyMongoClient.CountCursor(cursor)                                         返回游标的文档总数
    # PyMongoClient.SortCursor(cursor, col_or_list, director='ASC')             游标排序，默认升序，取值ASC/DESC
    # PyMongoClient.SubCursor(cursor, limit, skip=0)                            游标截取
    # ---------------------------------------------------------------------------------------------------
    # PyMongoClient.Aggregate(collection, pipleline)                            聚合
    # PyMongoClient.RunCommand(collection, cmdObj)                              运行数据库命令
    # ---------------------------------------------------------------------------------------------------
    # PyMongoClient.Str2ObjectId(id_str)                                        id字符串转id对象
    # PyMongoClient.ObjectId2Str(id_obj)                                        id对象转id字符串
    # PyMongoClient.GetBinaryFromFile(sourceFile)                               读文件，返回二进制内容
    # PyMongoClient.SaveBinaryToFile(binary, targetFile)                        将二进制内容保存为文件
    # ---------------------------------------------------------------------------------------------------
    # PyMongoClient.PutFile(localFilePath, dbFileName=None)                     将文件保存到GridFS并返回FileId
    # PyMongoClient.GetFile(fileId, localFilePath)                              将文件从GridFS取出，并保存到文件中
    # PyMongoClient.GetFilesCursor(docFilter=None, colFilter=None)              取得文件信息游标
    # PyMongoClient.DeleteFile(fileId)                                          删除GridFS中的文件

MongoDB,SQL常用语句对比

    # 1. 查询(find) ---------------------------------------------------------------------------------------------------
        查询所有结果
        select * from article
        db.article.find()

        指定返回哪些键
        select title, author from article
        db.article.find({}, {"title": 1, "author": 1})

        where条件
        select * from article where title = "mongodb"
        db.article.find({"title": "mongodb"})

        and条件
        select * from article where title = "mongodb" and author = "god"
        db.article.find({"title": "mongodb", "author": "god"})

        or条件
        select * from article where title = "mongodb" or author = "god"
        db.article.find({"$or": [{"title": "mongodb"}, {"author": "god"}]})

        比较条件
        select * from article where read >= 100;
        db.article.find({"read": {"$gt": 100}})

        > $gt(>)、$gte(>=)、$lt(<)、$lte(<=)
         select * from article where read >= 100 and read <= 200
         db.article.find({"read": {"$gte": 100, "lte": 200}})

        in条件
        select * from article where author in ("a", "b", "c")
        db.article.find({"author": {"$in": ["a", "b", "c"]}})

        like
        select * from article where title like "%mongodb%"
        db.article.find({"title": /mongodb/})

        count
        select count(*) from article
        db.article.count()

        不等于
        select * from article where author != "a"
        db.article.find({ "author": { "$ne": "a" }})

        排序

            升序：
            select * from article where type = "mongodb" order by read desc
            db.article.find({"type": "mongodb"}).sort({"read": -1})

            降序：
            select * from article where type = "mongodb" order by read asc
            db.article.find({"type": "mongodb"}).sort({"read": 1})
            findOne()：除了只返回一个查询结果外，使用方法与find()一样。

    # 2.创建（insert） ---------------------------------------------------------------------------------------------------
        insert into article(title, author, content) values("mongodb", "tg", "haha")
        db.article.insert({"title": "mongodb", "author": "tg", "content": "haha"})

    # 3.更新（update） ---------------------------------------------------------------------------------------------------
        update()
        语法：
            db.collecion.update(query, update[, options] )
            query : 必选，查询条件，类似find中的查询条件。
            update : 必选，update的对象和一些更新的操作符（如$,$inc...）等
            options：可选，一些更新配置的对象。
            upsert：可选，这个参数的意思是，如果不存在update的记录，是否插入objNew,true为插入，默认是false，不插入。
            multi：可选，mongodb 默认是false,只更新找到的第一条记录，如果这个参数为true,就把按条件查出来多条记录全部更新。
            writeConcern：可选，抛出异常的级别。

        简单更新：
        update article set title = "mongodb" where read > 100
        db.article.update({"read": {"$gt": 100}}, {"$set": { "title": "mongodb"}})
        save()
        db.article.save({_id: 123, title: "mongodb"})
        执行上面的语句，如果集合中已经存在一个_id为123的文档，则更新对应字段;否则插入。
        注：如果更新对象不存在_id，系统会自动生成并作为新的文档插入。

        更新操作符

            更新特定字段（$set）：
            update game set count = 10000 where _id = 123
            db.game.update({"_id": 123}, { "$set": {"count": 10000}})

            删除特定字段（$unset）：
            注：$unset指定字段的值只需是任意合法值即可。
            递增或递减（$inc）
             db.game.update({"_id": 123}, { "$inc": {"count": 10}}) // 每次count都加10
            > 注意：$inc对应的字段必须是数字，而且递增或递减的值也必须是数字。

            数组追加（$push）：
             db.game.update({"_id": 123}, { "$push": {"score": 123}})
            还可以一次追加多个元素：
             db.game.update({"_id": 123}, {"$push": {"score": [12,123]}})
            注：追加字段必须是数组。如果数组字段不存在，则自动新增，然后追加。
            一次追加多个元素（$pushAll）：
             db.game.update({"_id": 123}, {"$pushAll": {"score": [12,123]}})

            追加不重复元素（$addToSet）：
            $addToSet类似集合Set，只有当这个值不在元素内时才增加：
             db.game.update({"_id": 123}, {"$addToSet": {"score": 123}})

            删除元素（$pop）：
            db.game.update({"_id": 123}, {"$pop": {"score": 1}})  // 删除最后一个元素
            db.game.update({"_id": 123}, {"$pop": {"score": -1}})  // 删除第一个元素
            注：$pop每次只能删除数组中的一个元素，1表示删除最后一个，-1表示删除第一个。

            上面的语句表示删除数组score内值等于123的元素。
            删除多个特定元素（$pullAll）：
            db.game.update({"_id": 123}, {"$pullAll": {score: [123,12]}})

            上面的语句表示删除数组内值等于123或12的元素。
            更新嵌套数组的值：
            使用数组下标（从0开始）：
            {
                address: [{place: "nanji", tel: 123}, {place: "dongbei", tel: 321}]
            }
             db.game.update({"_id": 123}, {"$set": {"address.0.tel": 213}})
            如果你不知道要更新数组哪项，我们可以使用$操作符（ $表示自身，也就是按查询条件找出的数组里面的项自身，而且只会应用找到的第一条数组项）：
            在上面的语句中，$就是查询条件{"address.place": "nanji"}的查询结果，也就是{place: "nanji", tel: 123}，所以{"address.$.tel": 123}也就是{"address.{place: "nanji", tel: 123}.tel": 123}

    # 4. 删除（remove） ---------------------------------------------------------------------------------------------------
        删除所有文档：
        delete from article
        db.article.remove()

        删除指定文档：
        delete from article where title = "mongodb"
            db.article.remove({title: "mongodb"})
"""

import datetime
import os
import time

import pymongo

from data.config import DIR_dict
from lib.common import Common
from lib.yeyuc_logging import LoggingPython


# 连接管理
class Client():
    def __init__(self, db='test', **kwargs):
        self.loop = kwargs.get('loop', 5)  # 数据库失去连接后，尝试执行数据库操作的次数
        self.rate = float(kwargs.get('loop', 8))  # 数据库失去连接后，尝试执行数据库操作的时间间隔，首次尝试的间隔是rate的倒数，以后间隔时间增倍
        try:
            self.conn = pymongo.MongoClient(kwargs.get('host', 'localhost'), int(kwargs.get('port', 27017)))
            self.logger.info("\nMongoDB启动成功!\n")
            self.user, self.passwd = kwargs.get('user', 'root'), kwargs.get('password', '密码')
            self.SetDatabase('admin', user=self.user, passwd=self.passwd)
            self.SetDatabase(db, user=self.user, passwd=self.passwd)
        except Exception as errMsg:
            raise Exception(errMsg)

    # ---------------------------------------------------------------------------------------------------
    def SetDatabase(self, db, **kwargs):
        # 设置（修改）当前数据库
        self.db = self.conn[db]
        user, passwd = kwargs.get('user', self.user), kwargs.get('passwd', self.passwd)

        if db == 'admin':
            if not self.db.authenticate(user, passwd):
                raise Exception(u'数据库权限验证失败！')
        else:
            self.logger.info("\n已连接数据库{0}\n".format(db))

    def CloseConnection(self):
        # 关闭数据库连接
        self.conn.close()

    def Logout(self):
        # 注销用户
        self.db.logout()

    def GetStatus(self):
        # 获取MogoDB服务器的状态
        return self.db.last_status()

    def IsMongos(self):
        # 判断是否是MongoS
        return self.conn.is_mongos

    def GetDateTime(self):
        # 获取MongoDB服务器的当前时间（需要权限支持，若无权限，则返回本地时间）
        for i in range(self.loop):
            try:
                return self.db.eval("return new Date();")
            except pymongo.errors.AutoReconnect:
                time.sleep(pow(2, i) / self.rate)
            except Exception as e:
                return datetime.datetime.now()

        raise Exception(u'重连数据库失败！')


# 集合管理
class CollectionClient():
    def GetCollectionsDict(self):
        # 获取当前数据库的全部集合
        try:
            return {
                cls: self.db[cls].count()
                for cls in self.db.collection_names()
            }
        except Exception as e:
            print(e)

    def GetCollections(self):
        # 获取当前数据库的全部集合
        for i in range(self.loop):
            try:
                return self.db.collection_names()
            except pymongo.errors.AutoReconnect:
                time.sleep(pow(2, i) / self.rate)
        raise Exception(u'重连数据库失败！')

    def CreateCollection(self, collection):
        # 在当前数据库内创建新的集合
        self.logger.warning("正在创建集合{0}".format(collection))
        try:
            self.db.create_collection(collection)
            self.logger.info("集合{0}已创建！".format(collection))
        except Exception as e:
            self.logger.warning("集合创建失败，原因：{0}".format(e))

    def DropCollection(self, collection):
        # 删除当前数据库内名为collection的集合
        self.logger.warning("正在删除集合{0}".format(collection))
        try:
            self.db.drop_collection(collection)
            self.logger.info("集合{0}已被删除！".format(collection))
        except Exception as e:
            self.logger.warning("集合删除失败，原因：{0}".format(e))

    def ImportCollection(self, collection, name, host, **kwargs):
        """
        往当前数据库中导入文件,形成collection
        :param collection: 集合名
        :param name: 文件名
        :param kwargs: path: DIR_dict.get('JSON_DIR')
        :return: None
        """

        path = kwargs.get('path', DIR_dict.get('JSON_DIR'))
        file = os.path.join(path, name + '.json')
        USER = kwargs.get('USER', 'root')
        PASSWORD = kwargs.get('PASSWORD', '密码')

        self.logger.warning("正在往当前数据库中导入json文件,形成collection：{0}".format(collection))
        try:
            # os.system("""mongoimport -h 192.168.0.253 --authenticationDatabase admin -uroot -p 密码 -d {0} -c {1} --file {2}""".format(self.db.name, collection, file))
            os.system(
                """mongoimport -h {0} --authenticationDatabase admin -u {1} -p {2} -d {3} -c {4} --file {5}""".format(
                    host, USER, PASSWORD, self.db.name, collection, file))
        except Exception as e:
            self.logger.warning("集合导入失败，原因：{0}".format(e))

    def ExportCollection(self, collection, name, host='127.0.0.1', **kwargs):
        """
        往当前数据库中导出文件,形成collection
        :param collection: 集合名
        :param name: 文件名
        :param kwargs: path: DIR_dict.get('JSON_DIR')
        :return: None
        """

        path = kwargs.get('path', DIR_dict.get('JSON_DIR'))
        file = os.path.join(path, name + '.json')
        USER = kwargs.get('USER', self.user)
        PASSWORD = kwargs.get('PASSWORD', self.passwd)

        self.logger.warning("正在将当前数据库的collection：{0}导出,形成json文件".format(collection))
        try:
            # os.system("""mongoexport -h {0} -d {1} -c {2} -o {3}""".format(host, self.db.name, collection, file))
            os.system(
                """mongoexport -h {0} --authenticationDatabase admin -u {1} -p {2} -d {3} -c {4} -o {5}""".format(host,
                                                                                                                  USER,
                                                                                                                  PASSWORD,
                                                                                                                  self.db.name,
                                                                                                                  collection,
                                                                                                                  file))
        except Exception as e:
            self.logger.warning("集合导出失败，原因：{0}".format(e))

    def ExportDatabase(self, db, host):
        self.SetDatabase(db)
        cols = self.GetCollections()
        print("待导出集合：{0}\n".format(cols))
        for i, col in enumerate(cols):
            name = db + '\\' + col
            self.ExportCollection(col, name, host)
            print("{0}/{1}：集合{2}导出完毕！".format(i + 1, len(cols), col))
        print("全部集合导出完毕！")

    def ImportDatabase(self, db, host, **kwargs):
        path = kwargs.get('path', os.path.join(DIR_dict.get('JSON_DIR'), db))
        cols = [col.split('.json')[0] for col in os.listdir(path)]
        print("待导入集合：{0}\n".format(cols))
        for i, col in enumerate(cols):
            name = db + '\\' + col
            self.ImportCollection(col, name, host)
            print("{0}/{1}：集合{2}导入完毕！".format(i + 1, len(cols), col))
        print("全部集合导入完毕！")

        if not kwargs.get('save_dir'):
            self.common.rmdir(path)

    def Backup(self, db, host, **kwargs):

        path = kwargs.get('path', DIR_dict.get('BACKUP_DIR'))
        file = os.path.join(path, 'mongo')

        USER = kwargs.get('USER', self.user)
        PASSWORD = kwargs.get('PASSWORD', self.passwd)

        print("正在备份数据库：{0}\n".format(db))
        try:
            os.system(
                """mongodump -h {0} --authenticationDatabase admin -u {1} -p {2} -d {3} -o {4}""".format(host, USER,
                                                                                                         PASSWORD, db,
                                                                                                         file))
        except Exception as e:
            self.logger.warning("数据库备份失败，原因：{0}".format(e))

        print("数据库{0}备份完毕！".format(db))

    def Restore(self, db, host, **kwargs):
        """
        mongorestore -h 172.39.215.213 --authenticationDatabase admin -uroot -p 密码 -d epo --dir /root/backup/mongo/epo
        :return:
        """
        path = kwargs.get('path', DIR_dict.get('BACKUP_DIR'))
        file = os.path.join(path, 'mongo/{0}'.format(db))

        USER = kwargs.get('USER', self.user)
        PASSWORD = kwargs.get('PASSWORD', self.passwd)

        print("正在恢复数据库：{0}\n".format(db))
        try:
            os.system(
                """mongorestore -h {0} --authenticationDatabase admin -u {1} -p {2} -d {3} --dir {4}""".format(host,
                                                                                                               USER,
                                                                                                               PASSWORD,
                                                                                                               db,
                                                                                                               file))
        except Exception as e:
            self.logger.warning("数据库恢复失败，原因：{0}".format(e))

        print("数据库{0}恢复完毕！".format(db))


# 索引管理
class IndexClient():
    def IndexInformation(self, collection):
        # 获取索引信息
        for i in range(self.loop):
            try:
                return self.db[collection].index_information().keys()
            except pymongo.errors.AutoReconnect:
                time.sleep(pow(2, i) / self.rate)
        raise Exception(u'重连数据库失败！')

    def EnsureIndex(self, collection, key_or_list, unique, **kwargs):
        # 检查索引是否存在，若不存在，则创建索引，若存在，返回None
        # list参数形如：[('start_time', pymongo.ASCENDING), ('end_time', pymongo.ASCENDING)]
        self.logger.info("正在往集合{0}中建立索引".format(collection))
        for i in range(self.loop):
            try:
                self.db[collection].ensure_index(key_or_list,
                                                 unique=unique,
                                                 background=kwargs.get("backgroud", False)
                                                 )
                self.logger.info("索引创建成功！")
                return True
            except Exception as e:
                self.logger.warning("索引创建失败，原因：{0}".format(e))
                return False

    def CreateIndex(self, collection, key_or_list, unique, **kwargs):
        # 创建索引（推荐使用EnsureIndex）
        for i in range(self.loop):
            try:
                self.db[collection].create_index(key_or_list,
                                                 unique=unique,
                                                 background=kwargs.get("backgroud", False)
                                                 )
                return
            # except pymongo.errors.AutoReconnect:
            except Exception as e:
                time.sleep(pow(2, i) / self.rate)
        raise Exception(u'重连数据库失败！')

    def EnsureIndexes(self, collection, indexes, uniques):
        if not uniques:
            uniques = [False] * len(indexes)
        try:
            for i in range(len(indexes)):
                self.EnsureIndex(collection=collection,
                                 key_or_list=indexes[i],
                                 unique=uniques[i])
                # self.CreateIndex(collection=collection, key_or_list=indexes[i], unique=uniques[i])
        except Exception as e:
            print(e)
        return True

    def DropIndex(self, collection, key=None):
        # 删除索引，key=None时删除全部索引（_id除外）
        for i in range(self.loop):
            try:
                if key:
                    self.db[collection].drop_index(key)
                else:
                    self.db[collection].drop_indexes()
                return
            except pymongo.errors.AutoReconnect:
                time.sleep(pow(2, i) / self.rate)
        raise Exception(u'重连数据库失败！')


# 文件操作
class FileClient():

    def GetBinaryFromFile(self, sourceFile):
        # 读文件，返回二进制内容
        # 适用于在文档中直接保存小于16M的小文件，若文件较大时，应使用GridFS
        try:
            fp = open(sourceFile, 'rb')
            return bson.Binary(fp.read())
        except:
            return False
        finally:
            fp.close()

    def SaveBinaryToFile(self, binary, targetFile):
        # 将二进制内容保存为文件
        try:
            fp = open(targetFile, 'wb')
            fp.write(binary)
            return True
        except:
            return False
        finally:
            fp.close()

    def Str2ObjectId(self, id_str):
        return bson.ObjectId(id_str)

    def ObjectId2Str(self, id_obj):
        return str(id_obj)

    def PutFile(self, localFilePath, dbFileName=None):
        '''
        向GridFS中上传文件，并返回文件ID
        @localFilePath  本地文件路径
        @dbFileName     保存到GridFS中的文件名，如果为None则使用本地路径中的文件名
        '''

        fs = gridfs.GridFS(self.db)
        fp = open(localFilePath, 'rb')
        if dbFileName == None:
            dbFileName = os.path.split(localFilePath)[1]
        id = fs.put(fp, filename=dbFileName, chunkSize=4 * 1024 * 1024)
        fp.close()
        return id

    def GetFile(self, fileId, localFilePath=None):
        '''
        根据文件ID从GridFS中下载文件
        @fileId         文件ID
        @localFilePath  要保存的本地文件路径
        '''

        if isinstance(fileId, str):
            fileId = self.Str2ObjectId(fileId)

        fs = gridfs.GridFS(self.db)
        if localFilePath:
            fp = open(localFilePath, 'wb')
            try:
                fp.write(fs.get(fileId).read())
                return True
            except:
                return False
            finally:
                fp.close()
        else:
            try:
                return fs.get(fileId).read()
            except:
                return False

    def GetFilesCursor(self, docFilter=None, colFilter=None):
        '''
        取得GridFS中文件的游标
        可以进行过滤或检索的字段名有
        _id         文件ID
        filename    文件名
        length      文件大小
        md5         md5校验码
        chunkSize   文件块大小
        uploadDate  更新时间
        '''

        return self.GetCursor('fs.files',
                              docFilter=docFilter,
                              colFilter=colFilter)

    def DeleteFile(self, fileId):
        '''
        根据文件ID从GridFS中删除文件
        @fileId         文件ID
        '''

        fs = gridfs.GridFS(self.db)
        fs.delete(fileId)


# 聚合操作
class AggregateClient():
    def Aggregate(self, collection, pipleline):
        # 聚合
        # pipleline是一个由筛选、投射、分组、排序、限制、跳过等一系列构件组成管道队列
        for i in range(self.loop):
            try:
                return self.db[collection].aggregate(pipleline)
            except pymongo.errors.AutoReconnect:
                time.sleep(pow(2, i) / self.rate)
        raise Exception(u'重连数据库失败！')

    def RunCommand(self, collection, cmdObj):
        # 运行数据库命令
        # if cmdObj is a string, turns it into {cmdObj:1}
        for i in range(self.loop):
            try:
                return self.db[collection].runCommand(cmdObj)
            except pymongo.errors.AutoReconnect:
                time.sleep(pow(2, i) / self.rate)
        raise Exception(u'重连数据库失败！')


class MongoPython(Client, CollectionClient, IndexClient, FileClient, AggregateClient, LoggingPython):
    def __init__(self, **kwargs):
        LoggingPython.__init__(self, log_name="mongo")
        Client.__init__(self, db=kwargs.get("db", "test"), host=kwargs.get('host', '192.168.0.253'))
        self.common = Common()

    # 基本的增删查改
    def InsertDoc(self, collection, data, **kwargs):
        """
        :param 
            collection: 集合名
            data: 数据，list
            continue_on_error: 出错时继续
        :return 
            单条插入时返回单个id对象，
            批量插入时，返回id对象列表
        """
        try:
            res = self.db[collection].insert(data,
                                             manipulate=True,
                                             check_keys=True,
                                             continue_on_error=kwargs.get(
                                                 "continue_on_error", True))
            self.logger.info("全部文档已插入到{0}中！\n".format(collection))
            return res
        except pymongo.errors.DuplicateKeyError:
            self.logger.warning("<pymongo.errors.DuplicateKeyError>")
            self.logger.warning("逐一修改id，再插入到数据库！")
            i = self.CountDoc(collection)
            for doc in data:
                doc["_id"] = i
                self.db[collection].insert(doc,
                                           manipulate=True,
                                           check_keys=True,
                                           continue_on_error=kwargs.get(
                                               "continue_on_error", True))
                i += 1
            self.logger.info("全部文档已插入到{0}中！\n".format(collection))
            pass
        except Exception as e:
            ex_type, ex_val, ex_stack = sys.exc_info()
            self.logger.warning("文档插入过程出错，原因：{0}, 错误类型{1}".format(e, ex_type))
            return []

    def InsertBatchDoc(self, collection, data, **kwargs):
        """
        :param 
            collection: 集合名
            data: 数据，list
        :return 
            单条插入时返回单个id对象，
            批量插入时，返回id对象列表
        """
        self.logger.info("正在往集合{0}中批量插入文档...".format(collection))
        try:
            res = self.db[collection].insert_many(documents=data,
                                                  ordered=kwargs.get(
                                                      "order", True))
            self.logger.info("全部文档已插入到{0}中！\n".format(collection))
            return res
        except:
            self.InsertDoc(collection, data)

    def RemoveDoc(self, collection, docFilter=None):
        # 删除文档，docFilter=None时删除集合collection的全部文档
        for i in range(self.loop):
            try:
                return self.db[collection].remove(docFilter)
            except pymongo.errors.AutoReconnect:
                time.sleep(pow(2, i) / self.rate)
        raise Exception(u'重连数据库失败！')

    def UpdateDoc(self, collection, docFilter, data, **kwargs):
        # 更新文档，docFilter为更新对象的查找条件，data为更新数据，可以使用$inc/$set/$unset等修改器
        # 使用$set操作符将某个字段设置为指定值。
        # 操作符$inc可以为指定的键执行（原子）更新操作，如果字段存在，就将该值增加给定的增量，如果该字段不存在，就创建该字段。
        # $unset删除指定字段

        modifier = kwargs.get("modifier")
        for i in range(self.loop):
            try:
                if modifier:
                    return self.db[collection].update(docFilter, {modifier: data}, multi=kwargs.get("multi", True))
                else:
                    return self.db[collection].update(docFilter, data, multi=kwargs.get("multi", True))
            except pymongo.errors.AutoReconnect:
                time.sleep(pow(2, i) / self.rate)
        raise Exception(u'重连数据库失败！')

    def UpdateManyDocs(self, collection, docFilter, data, **kwargs):
        # 更新文档，docFilter为更新对象的查找条件，data为更新数据，可以使用$inc/$set/$unset等修改器
        # 使用$set操作符将某个字段设置为指定值。
        # 操作符$inc可以为指定的键执行（原子）更新操作，如果字段存在，就将该值增加给定的增量，如果该字段不存在，就创建该字段。
        # $unset删除指定字段

        modifier = kwargs.get("modifier", "$set")
        for i in range(self.loop):
            try:
                return self.db[collection].update_many(filter=docFilter,
                                                       update={modifier: data},
                                                       upsert=kwargs.get("upsert", False))
            except pymongo.errors.AutoReconnect:
                time.sleep(pow(2, i) / self.rate)
        raise Exception(u'重连数据库失败！')

    def GetDoc(self, collection, docFilter={}, sortFilter=None, **kwargs):
        """
        :param collection: 集合名
        :param docFilter: 过滤器
        :param projection: 返回字段
        :param sortFilter: 排序标准
        :param kwargs: 其他
        :return:

        普通查询：
            1、常用查询并排序
            db.getCollection('表名').find({"name" : "测试套餐1111"}).sort({"createDate":-1})
            注：1为升序，-1为降序

            2、多条件or查询
            db.getCollection('UserEntity').find({$or:[{"phone" : "18700000000"},{"phone" : "13400000000"}]})

            3、多条件and查询
            db.getCollection('UserEntity').find({"phone" : "18700000000","phone" : "13400000000"})

            4、模糊查询
            db.getCollection('UserEntity').find({"name" : /测试/})

            5、查询去重后name列数据
            db.getCollection('UserEntity').distinct("name")

            6、只查询5条数据
            db.getCollection('UserEntity').find().limit(5)

            7、查询5-10之间数据
            db.getCollection('UserEntity').find().limit(10).skip(5)

            8、查询记录条数
            db.getCollection('UserEntity').find().count()

            9、分组查询
            //单个字段分组查询
            db.getCollection('UserEntity').aggregate([{$group : {_id : "$balance", num : {$sum: 1}}}])

            //多个字段分组查询
            db.getCollection('UserEntity').aggregate([{$group : {_id : {balance:"$balance",expressInc:"$expressInc"}, num : {$sum : 1}}}])

        条件查询：
            > $gt , >= $gte, < $lt, <= $lte, != $ne
            db.tianyc02.find({age:{$lt:100,$gt:20}})

            $in ,$nin
            db.tianyc02.find({age:{$in:[11,22]}})
            db.tianyc02.find({age:{$nin:[11,22]}})

            注：gt=greater than  lt=less than   ne=not equal
        """
        # 返回单个文档
        projection = kwargs.get('projection', {'_id': False})
        if not kwargs.get("hide_log"):
            self.logger.info("正在查询集合{0}中的文档...".format(collection))

        try:
            if kwargs.get("limit"):
                if sortFilter:
                    result = self.db[collection].find(
                        filter=docFilter, projection=projection).sort(
                        sortFilter).limit(kwargs.get("limit"))
                else:
                    result = self.db[collection].find(
                        filter=docFilter,
                        projection=projection).limit(kwargs.get("limit"))
            else:
                if sortFilter:
                    result = self.db[collection].find(
                        filter=docFilter,
                        projection=projection).sort(sortFilter)
                else:
                    result = self.db[collection].find(
                        filter=docFilter, projection=projection)
            if not kwargs.get("hide_log"):
                self.logger.info("文档查询结束！")
            return result
        except Exception as e:
            self.logger.warning("文档查询过程出错，原因：{0}".format(e))

    def CountDoc(self, collection, docFilter=None):
        # 返回集合或查询的文档总数
        for i in range(self.loop):
            try:
                return self.db[collection].find(docFilter).count()
            except pymongo.errors.AutoReconnect:
                time.sleep(pow(2, i) / self.rate)
        raise Exception(u'重连数据库失败！')

    def GetCursor(self, collection, docFilter=None, colFilter=None):
        # 返回多个文档的游标
        for i in range(self.loop):
            try:
                if colFilter:
                    return self.db[collection].find(docFilter,
                                                    colFilter).batch_size(100)
                else:
                    return self.db[collection].find(docFilter).batch_size(100)
            except pymongo.errors.AutoReconnect:
                time.sleep(pow(2, i) / self.rate)
        raise Exception(u'重连数据库失败！')

    def CountCursor(self, cursor):
        # 返回游标的文档总数
        for i in range(self.loop):
            try:
                return cursor.count()
            except pymongo.errors.AutoReconnect:
                time.sleep(pow(2, i) / self.rate)
        raise Exception(u'重连数据库失败！')

    def SortCursor(self, cursor, col_or_list, director='ASC'):
        # 游标排序，默认ASCENDING（升序），取值ASC/DESC
        # col_or_list，列名或者是由(列名,方向)组成的列表
        if isinstance(col_or_list, list):
            args = []
            for col in col_or_list:
                if col[1] == 'ASC':
                    args.append((col[0], pymongo.ASCENDING))
                else:
                    args.append((col[0], pymongo.DESCENDING))
            for i in range(self.loop):
                try:
                    return cursor.sort(
                        args
                    )  # cursor.sort([("UserName",pymongo.ASCENDING),("Email",pymongo.DESCENDING)])
                except pymongo.errors.AutoReconnect:
                    time.sleep(pow(2, i) / self.rate)
            raise Exception(u'重连数据库失败！')
        else:
            if director == 'ASC':
                director = pymongo.ASCENDING
            else:
                director = pymongo.DESCENDING
            for i in range(self.loop):
                try:
                    return cursor.sort(
                        col_or_list, director
                    )  # director取值：pymongo.ASCENDING（升序）、pymongo.DESCENDING（降序）
                except pymongo.errors.AutoReconnect:
                    time.sleep(pow(2, i) / self.rate)
            raise Exception(u'重连数据库失败！')

    def SubCursor(self, cursor, limit, skip=0):
        # 截取游标
        for i in range(self.loop):
            try:
                if skip:
                    return cursor.skip(skip).limit(limit)
                else:
                    return cursor.limit(limit)
            except pymongo.errors.AutoReconnect:
                time.sleep(pow(2, i) / self.rate)
        raise Exception(u'重连数据库失败！')


if __name__ == '__main__':
    ############示例############

    # client = MongoPython()
    #
    # db = 'test'
    # col = 'a99'
    # datas = [{
    #     "author": "Mike",
    #     "title": "MongoDB is fun",
    #     "text": "and pretty easy too!",
    #     "date": datetime.datetime(2009, 11, 10, 10, 45)
    # }]
    #
    # filter = {"author": {"$eq": "Mike"}}
    # docFilter = {"author": {"$eq": "Mike"}}
    # sortFilter = [("date", -1)]
    #
    # key_or_list = [[("author", 1)], [("text", 1), ("title", -1)]]
    # uniques = [False] * 2
    #
    # file_name = 'a01'

    # 设置（修改）当前数据库
    # client.SetDatabase(db=db, user='', passwd='')

    # 集合操作 ---------------------------------------------------------------------------------------------------
    # print(client.GetCollections())                                    # 获取当前数据库的全部集合
    # print(client.GetCollectionsDict())                                # 获取当前数据库的全部集合与文档数量
    # client.CreateCollection(col)                                      # 创建集合
    # client.DropCollection(col)                                        # 删除集合
    # client.ImportCollection(col, file_name)                           # 导入集合
    # client.ExportCollection(col, file_name + '_')                     # 导出集合
    # client.ExportDatabase()  # 导出数据库
    # client.ExportCollection(col, file_name + '_')                     # 导出数据库

    # 索引操作 ---------------------------------------------------------------------------------------------------
    # client.CreateIndex(col, key_or_list[0], uniques[0])               # 创建索引
    # client.EnsureIndex(col, key_or_list[1], uniques[1])  # 检查索引是否存在，若不存在，则创建索引
    # client.EnsureIndexes(col, key_or_list, uniques)                   # 检查索引是否存在，若不存在，则创建索引

    # 文档操作 ---------------------------------------------------------------------------------------------------
    # client.RemoveDoc(col, docFilter=None)  # 删除文档，docFilter=None时删除集合的全部文档
    # client.InsertDoc(col, datas)                                      # data为字典时，单条插入，data为列表时，批量插入
    # client.InsertBatchDoc(col, datas)                                 # data为字典时，单条插入，data为列表时，批量插入
    # client.UpdateDoc(col, docFilter, datas, modifier=None)            # 更新文档，支持使用$inc/$set/$unset等修改器
    # print(client.CountDoc(col, docFilter=None))                       # 返回集合或查询的文档总数
    # client.GetDoc(col, docFilter=docFilter, sortFilter=sortFilter)  # 返回单个文档的游标
    # client.GetCursor(col, docFilter=None, colFilter=None)             # 返回多个文档的游标

    # print(client.IndexInformation(col))                               # 获取集合的索引信息
    # client.DropIndex(col, key=None)                                   # 删除索引，key=None时删除全部索引（_id除外）

    pass
