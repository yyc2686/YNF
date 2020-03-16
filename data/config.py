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
    # 'HOST': '192.168.1.110',
    'HOST': 'localhost',
    'PORT': '3306',
}

MONGODB = {
    'NAME': ['epo', 'export', 'uspto', 'wanfang'],
    # 'HOST': '192.168.1.110',
    'HOST': 'localhost',
    'PORT': 27017,
}

ONGING_PROJECT = [
    'bin.epo',
    'bin.export',
    'bin.uspto',
    'bin.wanfang',
]

EPO_CONFIG = {
    'DB': 'epo',
    'CTRY': ['PH', 'DO', 'KZ', 'LB', 'RE', 'SL', 'ME', 'SN', 'MT', 'IR', 'LU', 'LT', 'EC', 'TR', 'BN', 'DK', 'LR', 'YU',
             'RS', 'ML', 'FI', 'BZ', 'NO', 'TH', 'CN', 'PR', 'UZ', 'PT', 'JO', 'BG', 'VI', 'IS', 'LV', 'MA', 'VC', 'DE',
             'AZ', 'NZ', 'SU', 'HK', 'BB', 'QA', 'LI', 'CY', 'GB', 'LK', 'AU', 'TC', 'GA', 'UY', 'ZW', 'IT', 'VE', 'SA',
             'CK', 'EG', 'BH', 'WS', 'GT', 'BS', 'KW', 'SI', 'SE', 'ES', 'NL', 'MY', 'AT', 'AR', 'IN', 'BE', 'RH', 'CU',
             'GE', 'AN', 'KY', 'MX', 'LY', 'SC', 'CO', 'TN', 'PA', 'BY', 'GR', 'JP', 'MU', 'AD', 'IM', 'IL', 'RO', 'ID',
             'AE', 'UA', 'DM', 'GI', 'CZ', 'PL', 'SM', 'CR', 'SG', 'PE', 'MO', 'DZ', 'CH', 'EP', 'EE', 'IE', 'AG', 'HR',
             'US', 'MC', 'HU', 'VG', 'MH', 'FR', 'ZA', 'TW', 'KR', 'CL', 'BM', 'CS', 'VN', 'CA', 'DD', 'PK', 'WO', 'RU',
             'SY', 'BR', 'KP', 'CW', 'SK'],
    # 'EURO': {'AD': 'Andorra', 'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'BM': 'Bermuda', 'BY': 'Belarus',
    #          'CH': 'Switzerland', 'CS': 'Czechoslovakia', 'CW': 'Cornwall', 'CY': 'Cyprus', 'CZ': 'Czech Republic',
    #          'DD': 'German Democratic Republic', 'DE': 'Germany', 'DK': 'Denmark', 'EE': 'Estonia', 'EP': 'EPO',
    #          'ES': 'Spain', 'FI': 'Finland', 'FR': 'France', 'GB': 'United Kingdom', 'GI': 'Gibraltar', 'GR': 'Greece',
    #          'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland', 'IM': 'Isle of Man', 'IS': 'Iceland', 'IT': 'Italy',
    #          'LI': 'Liechtenstein', 'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia', 'MC': 'Monaco',
    #          'ME': 'Montenegro', 'MT': 'Malta', 'NL': 'Netherlands', 'NO': 'Norway', 'PL': 'Poland', 'PT': 'Portugal',
    #          'RE': 'Reunion', 'RO': 'Romania', 'RS': 'Serbia', 'RU': 'Russia', 'SE': 'Sweden', 'SI': 'Slovenia',
    #          'SK': 'Slovakia', 'SM': 'San Marino', 'TR': 'Turkey', 'UA': 'Ukraine', 'WO': 'WIPO', 'YU': 'Yugoslavia'},
    # 'NON-EURO': {'DZ': 'Algeria', 'EG': 'Egypt', 'GA': 'Gabon', 'LR': 'Liberia', 'LY': 'Libya', 'MA': 'Morocco',
    #              'ML': 'Mali', 'MU': 'Mauritius', 'RH': 'Rhodesia', 'SC': 'Seychelles', 'SL': 'Sierra Leone',
    #              'SN': 'Senegal', 'TN': 'Tunisia', 'ZA': 'South Africa', 'ZW': 'Zimbabwe', 'AG': 'Antigua and Barb.',
    #              'AR': 'Argentina', 'BB': 'Barbados', 'BR': 'Brazil', 'BS': 'Bahamas', 'BZ': 'Belize', 'CA': 'Canada',
    #              'CL': 'Chile', 'CO': 'Colombia', 'CR': 'Costa Rica', 'CU': 'Cuba', 'DM': 'Dominican Rep.',
    #              'EC': 'Ecuador', 'GT': 'Guatemala', 'KY': 'Cayman Is.', 'MX': 'Mexico', 'PA': 'Panama', 'PE': 'Peru',
    #              'PR': 'Puerto Rico', 'TC': 'Turks and Caicos Is.', 'US': 'United States', 'UY': 'Uruguay',
    #              'VC': 'St. Vin. and Gren.', 'VE': 'Venezuela', 'VG': 'Virgin Islands', 'AU': 'Australia',
    #              'CK': 'Cook Islands', 'MH': 'Marshall Islands', 'NZ': 'New Zealand', 'WS': 'Samoa', 'AN': 'Antilles',
    #              'DO': 'Dominica Rep.', 'VI': 'U.S. Virgin Is.', 'AE': 'Arab Emirates', 'AZ': 'Azerbaijan',
    #              'BH': 'Bahrain', 'BN': 'Brunei', 'CN': 'China', 'GE': 'Georgia', 'HK': 'Hong Kong,China',
    #              'ID': 'Indonesia', 'IL': 'Israel', 'IN': 'India', 'IR': 'Iran', 'JO': 'Jordan', 'JP': 'Japan',
    #              'KP': 'Dem. Rep. Korea', 'KR': 'Korea', 'KW': 'Kuwait', 'KZ': 'Kazakhstan', 'LB': 'Lebanon',
    #              'LK': 'Sri Lanka', 'MO': 'Macau', 'MY': 'Malaysia', 'PH': 'Philippines', 'PK': 'Pakistan',
    #              'QA': 'Qatar', 'SA': 'Saudi Arabia', 'SG': 'Singapore', 'SU': 'USSR', 'SY': 'Syria', 'TH': 'Thailand',
    #              'TW': 'Taiwan,China', 'UZ': 'Uzbekistan', 'VN': 'Vietnam'},
    'EURO': {'DE': 161931, 'FR': 59369, 'CH': 32865, 'IT': 26653, 'GB': 23806, 'NL': 21364, 'SE': 10454, 'AT': 8937,
             'BE': 8106, 'FI': 5533, 'ES': 4399, 'DK': 3802, 'LI': 1733, 'LU': 1343, 'NO': 1086, 'IE': 1026, 'PL': 566,
             'TR': 479, 'RU': 461, 'HU': 343, 'CZ': 337, 'PT': 204, 'SI': 199, 'MT': 161, 'GR': 158},
    'NON-EURO': {'JP': 221409, 'US': 160054, 'KR': 20929, 'CN': 7708, 'CA': 6535, 'AU': 2340, 'IL': 2218,
                 'SG': 668, 'VG': 655, 'IN': 511},
}

FINISHED_PROJECT = []

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

CONFIG_DICT = {
    # "pdfkit": pdfkit.configuration(wkhtmltopdf=DIR_dict["Wkhtmltopdf_DIR"]),
    # "imgkit": imgkit.config(wkhtmltoimage=DIR_dict["Wkhtmltoimg_DIR"]),
}
