import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DOC_DIR = os.path.join(BASE_DIR, "docs")
WHL_DIR = os.path.join(DOC_DIR, "whl")
whls = os.listdir(path=WHL_DIR)
whls = {whl.split("-")[0]: whl for whl in whls}

env_name = "patent"
try:
    os.system("pip install -i https://pypi.douban.com/simple virtualenvwrapper-win")
    os.system("mkvirtualenv {0}".format(env_name))
    os.system("workon {0}".format(env_name))
    os.system("python -m pip install --upgrade pip")
    with open("requirements.txt", "r") as f:
        lib = f.readline().strip()
        while lib:
            res = os.system("pip install -i https://pypi.douban.com/simple " + lib)
            if res == 0:
                print("\nSuccessful install {0}\n".format(lib))
            else:
                path = os.path.join(WHL_DIR, whls.get(lib.split("==")[0]))
                res = os.system("pip install " + path)
                if res == 0:
                    print("\nSuccessful install {0}\n".format(lib))
                else:
                    print("Failure Somehow!")
            lib = f.readline().strip()
except Exception as e:
    print(e)
