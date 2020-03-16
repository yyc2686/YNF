# -*- coding: utf-8 -*-

"""
概要：将multiprocessing的常用配置封装成一个类MultiCore

使用方法：

    步骤1：继承MultiCore, 实例化NewMultiCore，示例：client = NewMultiCore()
    步骤2：重构job函数，示例：
            >> def job(self)  # 无参
            >> def job(self, x)  # 单参
            >> def job(self, data):  # 多参
                   x, y = data[0], data[1]
    步骤3：进程池调用job函数进行运算，示例：
            >> client.process()  # 无参
            >> res = client.process([(1000000, 1)])  # 单参
            >> res = client.process([(1000000, 1), (1000000, 2)])  # 多参

拓展：

    使用共享内存，定义全局变量与数组
    使用锁，控制进程对于共享内存的访问
"""

import multiprocessing as mp


class MultiCore():
    def __init__(self, **kwargs):
        pass

    def job(self):
        pass

    def process(self, param=[], **kwargs):
        """
        :param param:
            self.job()函数的参数配置
        :param kwargs:
            processes: int, 核数，默认全部
            apply_async: bool, 方法选择，默认pool.map
        :return: 计算结果，list
        """
        pool = mp.Pool(processes=kwargs.get("processes"))  # 构建进程池
        if kwargs.get("apply_async"):
            res = [pool.apply_async(self.job, (i,)) for i in param]
            res = [res.get() for res in res]
        else:
            res = pool.map(self.job, param)
        return res

    def share_memory(self):
        """
        v: 单个变量，'i'代表整数
        a: 变量数组(只允许一维), 'i'代表整数
        :return:
        """
        v = mp.Value('i', 0)
        a = mp.Array('i', [1, 2, 3])
        return

    def lock(self, v, num, l):
        """
        :param v: 共享变量/数组
        :param num: 普通变量
        :param l: l = mp.Lock()
        :return: 保证共享变量安全，协调进程调度
        """
        l.acquire()
        for _ in range(10):
            v.value += num
            print(v.value)
        l.release()


class NewMultiCore(MultiCore):
    def __init__(self):
        MultiCore.__init__(self)

    def job(self, data):
        x, y = data[0], data[1]

        res = 0
        for j in range(y):
            for i in range(x):
                res += i + i ** 2 + i ** 3
        return res


if __name__ == '__main__':
    import time

    st = time.time()
    client = NewMultiCore()
    # res = client.process([1000000, 1000000])
    res = client.process([(1000000, 1), (1000000, 1)])
    et = time.time()
    print('结果：{0}\n用时：{1}'.format(sum(res), str(et - st)))
    pass
