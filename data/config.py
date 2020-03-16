# -*- coding: utf-8 -*-
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 工程目录

# 各级文件夹目录
BIN_DIR = os.path.join(BASE_DIR, 'bin')
DOCS_DIR = os.path.join(BASE_DIR, 'data')
LIB_DIR = os.path.join(BASE_DIR, 'lib')
TESTS_DIR = os.path.join(BASE_DIR, 'tests')

# 各类文件目录
DIR_dict = {
    "BACKUP_DIR": os.path.join(DOCS_DIR, 'backup'),
    "CSV_DIR": os.path.join(DOCS_DIR, 'csv'),
    "EXCEL_DIR": os.path.join(DOCS_DIR, 'excel'),
    "EXE_DIR": os.path.join(DOCS_DIR, 'exe'),
    "H5_DIR": os.path.join(DOCS_DIR, 'h5'),
    "JPG_DIR": os.path.join(DOCS_DIR, 'jpg'),
    "JSON_DIR": os.path.join(DOCS_DIR, 'json'),
    "LOG_DIR": os.path.join(DOCS_DIR, 'log'),
    "PDF_DIR": os.path.join(DOCS_DIR, 'pdf'),
    "PICKLE_DIR": os.path.join(DOCS_DIR, 'pickle'),
    "PNG_DIR": os.path.join(DOCS_DIR, 'png'),
    "PY_DIR": os.path.join(DOCS_DIR, 'py'),
    "RAR_DIR": os.path.join(DOCS_DIR, 'rar'),
    "RB_DIR": os.path.join(DOCS_DIR, 'recycle_bin'),
    "TXT_DIR": os.path.join(DOCS_DIR, 'txt'),
    "WHL_DIR": os.path.join(DOCS_DIR, 'whl'),
}

DATABASES = {
    'NAME': ['epo', 'export', 'uspto', 'wanfang'],
    'USER': 'root',
    'PASSWORD': '123',
    'HOST': 'localhost',
    'PORT': '3306',
}

MONGODB = {
    'NAME': ['epo', 'export', 'uspto', 'wanfang'],
    'HOST': 'localhost',
    'PORT': 27017,
}

ONGING_PROJECT = [
    'bin.epo',
    'bin.export',
    'bin.uspto',
    'bin.wanfang',
]

LIBS = [
    'lib.common',
    'lib.yeyuc_logging',
    'lib.yeyuc_keras',
    'lib.yeyuc_matplotlib',
    'lib.yeyuc_mongo',
    'lib.yeyuc_multicore',
    'lib.yeyuc_mysql',
    'lib.yeyuc_networkx',
    'lib.yeyuc_read',
    'lib.yeyuc_sklearn',
    'lib.yeyuc_spider',
    'lib.yeyuc_tensorflow',
    'lib.yeyuc_write',
]

import matplotlib.pyplot as plt

COLOR_DICT = {
    'tab20c': plt.get_cmap('tab20c'),
}

FONT_DICT = {
    'font': {'family': 'Times New Roman', 'weight': 'normal', 'size': 16},
    'title': {'family': 'Times New Roman', 'weight': 'normal', 'size': 16},
    'axis': {'family': 'Times New Roman', 'weight': 'normal', 'size': 13},
    'legend': {'family': 'Times New Roman', 'weight': 'normal', 'size': 10},
    'sub_font': {'family': 'Times New Roman', 'weight': 'normal', 'size': 14},
    'sub_title': {'family': 'Times New Roman', 'weight': 'normal', 'size': 14},
    'sub_axis': {'family': 'Times New Roman', 'weight': 'normal', 'size': 11},
    'sub_legend': {'family': 'Times New Roman', 'weight': 'normal', 'size': 8},
}
