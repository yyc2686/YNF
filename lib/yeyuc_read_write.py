# -*- coding: utf-8 -*-

# @Time    : 2020/2/4
# @Author  : yeyuc
# @Email   : yeyucheng_uestc@163.com
# @File    : yeyuc_read_write.py
# @Software: PyCharm

"""
概要：整理python常用的文件类型的读写函数，并封装成类ReadWrite

使用方法：

    步骤1：实例化ReadWrite，示例：client = ReadWrite()
    步骤2：确定目标文件类型，调用相应读写函数完成任务

常用函数：

    # .csv文件 ---------------------------------------------------------------------------------------------------
    # client.read_csv(name)
    # client.write_csv(data, name)
    # .xlsx文件 --------------------------------------------------------------------------------------------------
    # client.read_excel(name)
    # client.write_excel(data, name)
    # .h5文件 ----------------------------------------------------------------------------------------------------
    # client.read_h5(name)
    # client.write_h5(data, name)
    # .jpg文件 ----------------------------------------------------------------------------------------------------
    # client.read_jpg(name)
    # .json文件 ----------------------------------------------------------------------------------------------------
    # client.read_json(name)
    # client.write_json(data, name)
    # .pdf文件 ----------------------------------------------------------------------------------------------------
    # client.write_pdf(name)
    # .pickle文件 ----------------------------------------------------------------------------------------------------
    # client.read_pickle(name)
    # client.write_pickle(data, name)
    # .png文件 ----------------------------------------------------------------------------------------------------
    # client.read_png(name)
    # client.write_png(name)
    # .txt文件 ----------------------------------------------------------------------------------------------------
    # client.read_txt(name)
    # client.write_txt(data, name)
"""

import json
import os
import pickle

import pandas as pd

from data.config import DIR_dict


class CSV():

    def read_csv(self, name, **kwargs):
        """
        读取csv文件
        :param name: csv文件名
        :param kwargs: path，默认DIR_dict.get('CSV_DIR')
        :return: data, list/DataFrame
        """

        path = kwargs.get('path', DIR_dict.get('CSV_DIR'))
        name = name + '.csv'
        file = os.path.join(path, name)

        try:
            data = pd.read_csv(file)
            return data
        except Exception as e:
            print(e)

    def write_csv(self, data, name, **kwargs):
        """
        :param data: 数据，dict
        :param name: csv文件名
        :param kwargs: path，默认DIR_dict.get('CSV_DIR')
        :return: None
        """

        path = kwargs.get('path', DIR_dict.get('CSV_DIR'))
        name = name + '.csv'
        file = os.path.join(path, name)

        try:
            df = pd.DataFrame.from_dict(data,
                                        orient=kwargs.get('orient', 'columns'),
                                        dtype=kwargs.get('dtype'),
                                        columns=kwargs.get('columns')
                                        )
            df.to_csv(file, sep=',',
                      index=kwargs.get('index', False),
                      header=kwargs.get('header', True),
                      encoding=kwargs.get('encoding'),
                      )
        except Exception as e:
            print(e)


class Excel():

    def read_excel(self, name, **kwargs):
        """
        读取excel文件
        :param name: excel文件名
        :param kwargs: path，默认DIR_dict.get('CSV_DIR')
        :return: data, list/DataFrame
        """

        path = kwargs.get('path', DIR_dict.get('EXCEL_DIR'))
        name = name + '.xlsx'
        file = os.path.join(path, name)

        try:
            return pd.read_excel(file, )
        except Exception as e:
            print(e)
            return

    def write_excel(self, data, name, **kwargs):
        """
        :param data: 数据，dict
        :param name: excel文件名
        :param kwargs: path，默认DIR_dict.get('CSV_DIR')
        :return: None
        """

        path = kwargs.get('path', DIR_dict.get('EXCEL_DIR'))
        name = name + '.xlsx'
        file = os.path.join(path, name)

        try:
            df = pd.DataFrame.from_dict(data,
                                        orient=kwargs.get('orient', 'columns'),
                                        dtype=kwargs.get('dtype'),
                                        columns=kwargs.get('columns')
                                        )
            writer = pd.ExcelWriter(file)
            df.to_excel(writer,
                        sheet_name=kwargs.get('sheet_name', "Sheet1"),
                        index=kwargs.get('index', False),
                        header=kwargs.get('header', True),
                        encoding=kwargs.get('encoding'),
                        )
            writer.save()
        except Exception as e:
            print(e)


class H5():

    def read_h5(self, name, **kwargs):
        """
        读取h5文件
        :param name: h5文件名
        :param kwargs: path，默认DIR_dict.get('H5_DIR')
        :return: data, list/DataFrame
        """
        import h5py

        path = kwargs.get('path', DIR_dict.get('H5_DIR'))
        name = name + '.h5'
        file = os.path.join(path, name)

        try:
            data = h5py.File(file, 'r')
            return data
        except Exception as e:
            print(e)

    def write_h5(self, data, labels, name, **kwargs):
        """
        :param data: 数据，dict
        :param name: h5文件名
        :param kwargs: path，默认DIR_dict.get('H5_DIR')
        :return: None
        """
        import h5py

        path = kwargs.get('path', DIR_dict.get('H5_DIR'))
        name = name + '.h5'
        file = os.path.join(path, name)

        try:
            with h5py.File(file, 'w') as f:
                f['data'] = data  # 将数据写入文件的主键data下面
                f['labels'] = labels  # 将数据写入文件的主键labels下面
        except Exception as e:
            print(e)


class JPG():

    def read_jpg(self, name, **kwargs):
        """
        :param name: jpg文件名
        :param kwargs: path，默认DIR_dict.get('JPG_DIR')
        """
        from PIL import Image
        path = kwargs.get('path', DIR_dict.get('JPG_DIR'))
        name = name + '.jpg'
        file = os.path.join(path, name)
        try:
            im = Image.open(file)
            im.show()
        except Exception as e:
            print(e)


class Json():
    def read_json(self, name, **kwargs):
        """
        读取json文件
        :param name: json文件名
        :param kwargs: path，默认DIR_dict.get('JSON_DIR')
        :return: data, list/DataFrame
        """

        path = kwargs.get('path', DIR_dict.get('JSON_DIR'))
        name = name + '.json'
        file = os.path.join(path, name)

        try:
            return json.load(file)
        except:
            try:
                with open(file, 'rb') as f:
                    content = f.read().decode("utf-8")
                    return json.loads(content)
            except Exception as e:
                print(e)

    def write_json(self, data, name, **kwargs):
        """
        :param data: 数据，dict
        :param name: json文件名
        :param kwargs: path，默认DIR_dict.get('JSON_DIR')
        :return: None
        """

        path = kwargs.get('path', DIR_dict.get('JSON_DIR'))
        name = name + '.json'
        file = os.path.join(path, name)

        try:
            with open(file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            print(e)


class PDF():

    def write_pdf(self, name, **kwargs):
        """
        :param data: 数据，任意格式
        :param name: pdf文件名
        :param kwargs: path，默认DIR_dict.get('PDF_DIR')
        :return: None
        """

        path = kwargs.get('path', DIR_dict.get('PDF_DIR'))
        name = name + '.pdf'
        file = os.path.join(path, name)

        try:
            self.fig.savefig(file, dpi=500, bbox_inches='tight')
        except Exception as e:
            print(e)


class Pickle():
    def read_pickle(self, name, **kwargs):
        """
        读取pickle文件
        :param name: pickle文件名
        :param kwargs: path，默认DIR_dict.get('PICKLE_DIR')
        :return: data, 任意格式
        """

        path = kwargs.get('path', DIR_dict.get('PICKLE_DIR'))
        name = name + '.pickle'
        file = os.path.join(path, name)

        try:
            with open(file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(e)

    def write_pickle(self, data, name, **kwargs):
        """
        :param data: 数据，任意格式
        :param name: pickle文件名
        :param kwargs: path，默认DIR_dict.get('PICKLE_DIR')
        :return: None
        """

        path = kwargs.get('path', DIR_dict.get('PICKLE_DIR'))
        name = name + '.pickle'
        file = os.path.join(path, name)

        try:
            with open(file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(e)


class PNG():

    def read_png(self, name, **kwargs):
        """
        :param name: png文件名
        :param kwargs: path，默认DIR_dict.get('PNG_DIR')
        """
        from PIL import Image
        path = kwargs.get('path', DIR_dict.get('PNG_DIR'))
        name = name + '.png'
        file = os.path.join(path, name)
        try:
            im = Image.open(file)
            im.show()
        except Exception as e:
            print(e)

    def write_png(self, name, **kwargs):
        """
        :param data: 数据，任意格式
        :param name: png文件名
        :param kwargs: path，默认DIR_dict.get('PNG_DIR')
        :return: None
        """

        path = kwargs.get('path', DIR_dict.get('PNG_DIR'))
        name = name + '.png'
        file = os.path.join(path, name)

        try:
            self.fig.savefig(file, dpi=500, bbox_inches='tight')
        except Exception as e:
            print(e)


class TXT():
    def read_txt(self, name, **kwargs):
        """
        读取txt文件
        :param name: txt文件名
        :param kwargs: path，默认DIR_dict.get('TXT_DIR')
        :return: data, 任意格式
        """

        path = kwargs.get('path', DIR_dict.get('TXT_DIR'))
        name = name + '.txt'
        file = os.path.join(path, name)

        try:
            with open(file, 'r') as f:
                return f.readlines()
        except Exception as e:
            print(e)

    def write_txt(self, data, name, **kwargs):
        """
        :param data: 数据，任意格式
        :param name: txt文件名
        :param kwargs: path，默认DIR_dict.get('TXT_DIR')
        :return: None
        """

        path = kwargs.get('path', DIR_dict.get('TXT_DIR'))
        name = name + '.txt'
        file = os.path.join(path, name)

        try:
            with open(file, 'w') as f:
                for line in data:
                    f.write(line)
                    f.write('\n')
        except Exception as e:
            print(e)


class ReadWrite(CSV, Excel, H5, JPG, Json, PDF, Pickle, PNG, TXT):

    def __init__(self, **kwargs):
        pass


if __name__ == '__main__':
    client = ReadWrite()

    name = 'gdp'
    name = 'gdp_per_capita'
    c_info = client.read_json('{}_slice'.format(name))
    c_info = c_info.get('RECORDS')
    info = {}
    for data in c_info:
        info[data.get('year'), data.get('ctry_code')] = data.get(name)

    client.write_pickle(info, name)
    print(info)

    # c_info = client.read_json('c_info')
    # c_info = c_info.get('RECORDS')
    # info = {}
    # for data in c_info:
    #     # if data.get('Continent') == '欧洲':
    #     if data.get('Continent') != '欧洲':
    #         info[data.get('Id')] = data.get('Name')
    #     # info[data.get('Id')] = data.get('Name')
    # print(list(set(info)))

    # name = 'test'
    # data = {'name': ['Tom', 'Michael'], 'age': [12, 21]}
    # txt_data = ['hello', 'world']

    # .csv文件 ---------------------------------------------------------------------------------------------------
    # client.write_csv(data, name)
    # print(client.read_csv(name))
    # .xlsx文件 --------------------------------------------------------------------------------------------------
    # client.write_excel(data, name)
    # print(client.read_excel(name))
    # .h5文件 ----------------------------------------------------------------------------------------------------
    # client.write_h5(data, name)
    # client.read_h5(name)
    # .jpg文件 ----------------------------------------------------------------------------------------------------
    # client.read_jpg(name)
    # .pdf文件 ----------------------------------------------------------------------------------------------------
    # client.write_pdf(data, name)
    # .json文件 ----------------------------------------------------------------------------------------------------
    # client.write_json(data, name)
    # print(client.read_json(name))
    # .pickle文件 ----------------------------------------------------------------------------------------------------
    # client.write_pickle(data, name)
    # print(client.read_pickle(name))
    # .png文件 ----------------------------------------------------------------------------------------------------
    # client.write_png(name)
    # client.read_png(name)
    # .txt文件 ----------------------------------------------------------------------------------------------------
    # client.write_txt(txt_data, name)
    # print(client.read_txt(name))
    pass
