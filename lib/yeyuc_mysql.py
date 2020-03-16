# -*- coding: utf-8 -*-

# @Time    : 2020/2/3
# @Author  : yeyuc
# @Email   : yeyucheng_uestc@163.com
# @File    : yeyuc_mysql.py
# @Software: PyCharm

"""
概要：将MySQLdb的常用配置封装成一个类MysqlPython
使用方法：
    步骤1：实例化MysqlPython，示例：client = MysqlPython(db='test')
    步骤2：调用client的函数，完成增删查改等操作

常用函数：
    # 设置（修改）当前数据库
    client.SetDatabase(db)

    # 表单操作 ---------------------------------------------------------------------------------------------------
    # client.DropTable(table)  # 删表
    # client.CreateTable(create_sql)  # 建表
    # print(client.GetTables(db))  # 获取数据库表单信息

    # 索引操作 ---------------------------------------------------------------------------------------------------
    # for i in range(5):
    #     client.CreateIndex(table=table, type=types[i], content=contents[i])  # 创建索引
    # client.DropIndex(table, name='index_name')  # 删除索引

    # 记录操作 ---------------------------------------------------------------------------------------------------
    # client.InsertItem(table, items, datas)  # 插入记录
    # client.RemoveItem(table, delete_sql)  # 删除记录
    # print(client.GetItem(get_sql))  # 返回符合条件的记录
    # client.UpdateItem(update_sql)  # 更新记录

MongoDB,SQL常用语句对比

    # 1. 查询(find) ----------------------------------------------------------------------------------------------
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

    # 2.创建（insert） --------------------------------------------------------------------------------------------
        insert into article(title, author, content) values("mongodb", "tg", "haha")
        db.article.insert({"title": "mongodb", "author": "tg", "content": "haha"})

    # 3.更新（update） -------------------------------------------------------------------------------------------
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
        执行上面的语句，如果表中已经存在一个_id为123的记录，则更新对应字段;否则插入。
        注：如果更新对象不存在_id，系统会自动生成并作为新的记录插入。

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
            $addToSet类似表Set，只有当这个值不在元素内时才增加：
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

    # 4. 删除（remove） ------------------------------------------------------------------------------------------
        删除所有记录：
        delete from article
        db.article.remove()

        删除指定记录：
        delete from article where title = "mongodb"
            db.article.remove({title: "mongodb"})

23个MySQL常用查询语句：
    一查询数值型数据:
     SELECT * FROM tb_name WHERE sum > 100;
     查询谓词:>,=,<,<>,!=,!>,!<,=>,=<
     
    二查询字符串
     SELECT * FROM tb_stu  WHERE sname  =  '小刘'
     SELECT * FROM tb_stu  WHERE sname like '刘%'
     SELECT * FROM tb_stu  WHERE sname like '%程序员'
     SELECT * FROM tb_stu  WHERE sname like '%PHP%'
     
    三查询日期型数据
     SELECT * FROM tb_stu WHERE date = '2011-04-08'
     注:不同数据库对日期型数据存在差异: ：
     (1)MySQL:SELECT * from tb_name WHERE birthday = '2011-04-08'
     (2)SQL Server:SELECT * from tb_name WHERE birthday = '2011-04-08'
     (3)Access:SELECT * from tb_name WHERE birthday = #2011-04-08#
     
    四查询逻辑型数据
     SELECT * FROM tb_name WHERE type = 'T'
     SELECT * FROM tb_name WHERE type = 'F'
     逻辑运算符:and or not
     
    五查询非空数据
     SELECT * FROM tb_name WHERE address <>'' order by addtime desc
     注:<>相当于PHP中的!=
     
    六利用变量查询数值型数据
     SELECT * FROM tb_name WHERE id = '$_POST[text]' 
    注:利用变量查询数据时，传入SQL的变量不必用引号括起来，因为PHP中的字符串与数值型数据进行连接时，程序会自动将数值型数据转变成字符串，然后与要连接的字符串进行连接
     
    七利用变量查询字符串数据 
    SELECT * FROM tb_name WHERE name LIKE '%$_POST[name]%' 
    完全匹配的方法"%%"表示可以出现在任何位置
     
    八查询前n条记录
     SELECT * FROM tb_name LIMIT 0,N;
     limit语句与其他语句，如order by等语句联合使用，会使用SQL语句千变万化，使程序非常灵活
     
    九查询后n条记录
     SELECT * FROM tb_stu ORDER BY id ASC LIMIT n
     
    十查询从指定位置开始的n条记录
     SELECT * FROM tb_stu ORDER BY id ASC LIMIT _POST[begin],n
     注意:数据的id是从0开始的
     
    十一查询统计结果中的前n条记录
     SELECT * ,(yw+sx+wy) AS total FROM tb_score ORDER BY (yw+sx+wy) DESC LIMIT 0,$num
     
    十二查询指定时间段的数据
     SELECT  要查找的字段 FROM 表名 WHERE 字段名 BETWEEN 初始值 AND 终止值
     SELECT * FROM tb_stu WHERE age BETWEEN 0 AND 18
     
    十三按月查询统计数据
     SELECT * FROM tb_stu WHERE month(date) = '$_POST[date]' ORDER BY date ;
     注：SQL语言中提供了如下函数，利用这些函数可以很方便地实现按年、月、日进行查询
     year(data):返回data表达式中的公元年分所对应的数值
     month(data):返回data表达式中的月分所对应的数值
     day(data):返回data表达式中的日期所对应的数值
     
    十四查询大于指定条件的记录
     SELECT * FROM tb_stu WHERE age>$_POST[age] ORDER BY age;
     
    十五查询结果不显示重复记录
     SELECT DISTINCT 字段名 FROM 表名 WHERE 查询条件 
    注:SQL语句中的DISTINCT必须与WHERE子句联合使用，否则输出的信息不会有变化 ,且字段不能用*代替
     
    十六NOT与谓词进行组合条件的查询
     (1)NOT BERWEEN … AND … 对介于起始值和终止值间的数据时行查询 可改成 <起始值 AND >终止值
     (2)IS NOT NULL 对非空值进行查询 
     (3)IS NULL 对空值进行查询
     (4)NOT IN 该式根据使用的关键字是包含在列表内还是排除在列表外，指定表达式的搜索，搜索表达式可以是常量或列名，而列名可以是一组常量，但更多情况下是子查询
     
    十七显示数据表中重复的记录和记录条数
     SELECT  name,age,count(*) ,age FROM tb_stu WHERE age = '19' group by date
     
    十八对数据进行降序/升序查询
     SELECT 字段名 FROM tb_stu WHERE 条件 ORDER BY 字段 DESC 降序
     SELECT 字段名 FROM tb_stu WHERE 条件 ORDER BY 字段 ASC  升序
     注:对字段进行排序时若不指定排序方式，则默认为ASC升序
     
    十九对数据进行多条件查询
     SELECT 字段名 FROM tb_stu WHERE 条件 ORDER BY 字段1 ASC 字段2 DESC  …
     注意:对查询信息进行多条件排序是为了共同限制记录的输出，一般情况下，由于不是单一条件限制，所以在输出效果上有一些差别。
     
    二十对统计结果进行排序
     函数SUM([ALL]字段名) 或 SUM([DISTINCT]字段名),可实现对字段的求和，函数中为ALL时为所有该字段所有记录求和,若为DISTINCT则为该字段所有不重复记录的字段求和
     如：SELECT name,SUM(price) AS sumprice  FROM tb_price GROUP BY name
     
    SELECT * FROM tb_name ORDER BY mount DESC,price ASC
     
    二十一单列数据分组统计
     SELECT id,name,SUM(price) AS title,date FROM tb_price GROUP BY pid ORDER BY title DESC
     注:当分组语句group by排序语句order by同时出现在SQL语句中时，要将分组语句书写在排序语句的前面，否则会出现错误
     
    二十二多列数据分组统计
     多列数据分组统计与单列数据分组统计类似 
    SELECT *，SUM(字段1*字段2) AS (新字段1) FROM 表名 GROUP BY 字段 ORDER BY 新字段1 DESC
     SELECT id,name,SUM(price*num) AS sumprice  FROM tb_price GROUP BY pid ORDER BY sumprice DESC
     注：group by语句后面一般为不是聚合函数的数列，即不是要分组的列
     
    二十三多表分组统计
     SELECT a.name,AVG(a.price),b.name,AVG(b.price) FROM tb_demo058 AS a,tb_demo058_1 AS b WHERE a.id=b.id GROUP BY b.type;

"""

import MySQLdb

from lib.yeyuc_logging import LoggingPython


# 连接管理
class Client():
    def __init__(self, **kwargs):
        try:
            self.conn = MySQLdb.connect(host=kwargs.get('host', 'localhost'),
                                        user=kwargs.get('user', 'root'),
                                        password=kwargs.get('password', '123'),
                                        charset=kwargs.get('charset', 'utf8'),
                                        use_unicode=kwargs.get('use_unicode', True))
            self.cursor = self.conn.cursor()
            self.logger.info("\nMySQL连接成功!\n")
            if kwargs.get('db'):
                self.SetDatabase(kwargs.get('db'))
        except Exception as errMsg:
            raise Exception(errMsg)

    # ---------------------------------------------------------------------------------------------------
    def SetDatabase(self, db):
        try:
            self.cursor.execute('USE {0}'.format(db))  # 设置（修改）当前数据库
            self.logger.info("\n已连接数据库{0}\n".format(db))
        except Exception as e:
            self.logger.warning(e)

    def CloseConnection(self):
        # 关闭数据库连接
        self.conn.close()
        self.logger.info("\nMySQL连接已关闭!\n")


# 表单管理
class TableClient():

    def GetTables(self, db):
        # 获取当前数据库的全部表
        self.logger.info('\n正在获取数据库{0}的全部表单信息 ...'.format(db))
        self.cursor.execute(
            """SELECT TABLE_NAME, TABLE_ROWS FROM information_schema.tables where TABLE_SCHEMA='{0}' order by TABLE_NAME""".format(
                db))
        database_info = {item[0]: item[1] for item in self.cursor.fetchall()}
        if database_info:
            self.logger.info('查询成功！\n'.format(db))
        else:
            self.logger.warning('数据库{0}为空！\n'.format(db))
        return database_info

    def CreateTable(self, sql):
        # 在当前数据库内创建新的表
        self.logger.info("正在建表")
        try:
            self.cursor.execute(sql)
            self.logger.info("建表成功！")
            return True
        except Exception as e:
            self.logger.warning("表创建失败，原因：{0}".format(e))

    def DropTable(self, table):
        # 删除当前数据库内名为table的表
        self.logger.warning("正在删除表{0}".format(table))

        try:
            self.cursor.execute('DROP TABLE IF EXISTS {0}'.format(table))
            self.logger.info("表{0}已被删除！".format(table))
        except Exception as e:
            self.logger.warning("表删除失败，原因：{0}".format(e))


# 索引管理
class IndexClient():

    def DropIndex(self, table, name):
        """
        :param table: 表单, str
        :param name: 索引，默认index_开头，str
        :return: None
        """
        self.logger.info('开始删除表{0}中的索引{1} ...'.format(table, name))
        sql = """DROP INDEX {0} ON {1}; """.format(name, table)
        try:
            self.cursor.execute(sql)
            self.conn.commit()
            self.logger.info('索引删除完毕！\n')
        except Exception as e:
            self.logger.warning('索引删除失败，原因：{0}\n'.format(e))

    def CreateIndex(self, table, content, name='', type='NORMAL'):
        """
        :param table: 表单, str
        :param name: 索引, str
        :param content: 字段, tuple
        :param type: 索引类型，默认'NORMAL'/'', 可选：'UNIQUE', 'Full Text'
        :return: None
        """
        self.logger.info('开始往表{0}中添加索引 ...'.format(table))
        name = name if name else 'index_' + '_'.join(content)
        if len(content) == 1:
            sql = """CREATE {0} INDEX {1} ON {2}({3})""".format(type, name, table, content[0])
        elif len(content) > 1:
            sql = """CREATE {0} INDEX {1} ON {2}{3}""".format(type, name, table, tuple(content))
            sql = sql.replace('\'', '')

        try:
            self.cursor.execute(sql)
            self.conn.commit()
            self.logger.info('索引添加完毕！\n')
        except Exception as e:
            self.logger.warning('索引添加失败，原因：{0}\n'.format(e))


class MysqlPython(Client, TableClient, IndexClient, LoggingPython):
    def __init__(self, **kwargs):
        LoggingPython.__init__(self, log_name="mysql")
        Client.__init__(self, db=kwargs.get('db', 'testdb'))
        self.COMMIT_THRESHOLD = 10000

    # 基本的增删查改
    def InsertItem(self, table, items, records, type='IGNORE', **kwargs):
        """
        :param table: 表单名, str
        :param items: 字段名，tuple, 元素为str
        :param records: 记录，list, 元素为tuple
        :param type: 插入的方式：默认IGNORE
        :param kwargs: hide_log, 是否隐藏输出提示信息，默认不隐藏，反复调用时建议隐藏
        :return: None
        """
        values = tuple(['%s'] * len(items))
        sql = """
            INSERT {0} INTO {1}{2}
            VALUES {3}
        """.format(type, table, items, values)
        sql = sql.replace('\'', '')

        if not kwargs.get('hide_log'):
            self.logger.info('开始往表{0}中插入记录 ...'.format(table))
        count = 0
        try:
            for record in records:
                self.cursor.execute(sql, record)
                count += 1
                if count % self.COMMIT_THRESHOLD == 0:  # 单次插入上限
                    self.conn.commit()
            self.conn.commit()
            if not kwargs.get('hide_log'):
                self.logger.info('全部记录已插入到表{0}中！\n'.format(table))
        except Exception as e:
            self.logger.warning('记录插入失败，原因：{0}\n'.format(e))
            self.cursor.rollback()  # 发生错误时回滚

    def RemoveItem(self, table, sql):
        self.logger.warning("正在执行SQL删除语句：{0}".format(sql))
        try:
            self.cursor.execute(sql)
            self.conn.commit()
            self.logger.info('目标记录已全部从表{0}中删除！\n'.format(table))
        except Exception as e:
            self.logger.warning('记录删除失败，原因：{0}\n'.format(e))
            self.cursor.rollback()  # 发生错误时回滚

    def GetItem(self, sql):
        """
        :param sql: 查询语句
        :return: tuples
        """
        self.logger.warning("开始执行查询语句：{0}".format(sql))
        try:
            self.cursor.execute(sql)
            self.logger.info('目标记录查询完毕！\n')
            res = self.cursor.fetchall()
            return res
        except Exception as e:
            self.logger.warning('记录查询失败，原因：{0}\n'.format(e))

    def UpdateItem(self, sql):
        self.logger.warning("正在执行SQL更新语句：{0}".format(sql))
        try:
            self.cursor.execute(sql)
            self.conn.commit()
            self.logger.info('目标记录已全部更新！\n'.format(table))
        except Exception as e:
            self.logger.warning('记录更新失败，原因：{0}\n'.format(e))
            self.cursor.rollback()  # 发生错误时回滚


if __name__ == '__main__':
    client = MysqlPython()

    db = 'testdb'
    table = 'person'

    create_sql_1 = """
        CREATE TABLE `{0}` (
      `name` varchar(50) NOT NULL,
      `sex` varchar(10) NOT NULL,
      `ID` int(50) NOT NULL,
      `hobby` varchar(255) NOT NULL,
      PRIMARY KEY (`ID`),
      UNIQUE KEY `index_ID` (`ID`) USING BTREE,
      UNIQUE KEY `index_name_sex` (`name`,`sex`) USING BTREE,
      KEY `index_name` (`name`) USING BTREE
    )
    """.format(table)
    create_sql = """
        CREATE TABLE `{0}` (
      `name` varchar(50) NOT NULL,
      `sex` varchar(10) NOT NULL,
      `ID` int(50) NOT NULL,
      `hobby` varchar(255) NOT NULL,
      PRIMARY KEY (`ID`)
    )
    """.format(table)

    types = ['', '', 'UNIQUE', 'FULLTEXT', '']
    contents = [['name'], ['sex'], ['ID'], ['hobby'], ['name', 'sex']]

    items = ('name', 'sex', 'ID', 'hobby')
    datas = [('Tom', 'man', 123, 'walk'), ('Michael', 'man', 456, 'sleep')]

    delete_sql = """DELETE FROM {0} WHERE NAME = '{1}'""".format(table, 'Tom')
    get_sql = """SELECT * FROM {0} WHERE ID > 10000 ORDER BY ID ASC LIMIT {1}""".format(table, 10)
    update_sql = """UPDATE {0} SET SEX = '男' WHERE SEX = 'man'""".format(table)

    # 设置（修改）当前数据库
    client.SetDatabase(db)

    # 表单操作 ---------------------------------------------------------------------------------------------------
    # client.DropTable(table)  # 删表
    # client.CreateTable(create_sql)  # 建表
    # print(client.GetTables(db))  # 获取数据库表单信息

    # 索引操作 ---------------------------------------------------------------------------------------------------
    # for i in range(5):
    #     client.CreateIndex(table=table, type=types[i], content=contents[i])  # 创建索引
    # client.DropIndex(table, name='index_name')  # 删除索引

    # 记录操作 ---------------------------------------------------------------------------------------------------
    # client.InsertItem(table, items, datas)  # 插入记录
    # client.RemoveItem(table, delete_sql)  # 删除记录
    # print(client.GetItem(get_sql))  # 返回符合条件的记录
    # client.UpdateItem(update_sql)  # 更新记录
    pass
