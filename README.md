# -AI Creative Program-
Hello Paddle
# [AI训练营]paddleclas实现图像分类baseline

## 0 项目背景
> [课程链接：飞桨领航团AI达人创造营](https://aistudio.baidu.com/aistudio/course/introduce/24607?directly=1&shared=1)    

<hr>
<center><font size="3px" color="green">暑    期    充    电    季</font></center>
<hr>
<center><font size="2px" color="red">百度飞桨领航团全新推出“AI达人创造营”</font></center>
<center><font size="2px" color="green"><strong>十位飞桨开发者技术专家（PPDE）</strong>手把手教大家完成项目从idea思考到部署落地的全流程实战</font></center>
<center><font >最终让每位参与者都有一个可以给自己简历加分的项目</font></center>
<center><font >7月26日-8月16日，每晚 19:00-21:00 直播讲解、十位飞桨开发者技术专家（PPDE）手把手教助你成为AI达人</font></center>   
<center><font size="2px" color="blue">报名后，请加入课程 QQ 群861942585，QQ群用于直播提醒、实时答疑、交流互动等</font></center>  

<hr>
<font size="2px" color="red">PS:</font>

<font size="3px" color="black">本项目属于本次课程的大作业的一部分，希望大家可以学会使用paddleclas实现图像分类。</font>
<hr>

## 本大作业任务

1、选择一个心仪的数据集     
2、运行项目，<font color="red">能够跑通项目即可达到结业要求</font>     
3、记得生成新版本，公开项目哦~

**加分项：**     
4、更换网络、进行数据预处理和调参   

## 1 项目实现简介
正如标题，采用paddleclas套件实现分类[30分钟玩转PaddleClas（尝鲜版）](https://gitee.com/paddlepaddle/PaddleClas/blob/release/2.2/docs/zh_CN/tutorials/quick_start_new_user.md)    
查看套件，可以知道    
实现分类，仅仅需要我们将数据集提取为如下这种格式的txt文件即可（当然我们并不需要更大的训练集）    
![](https://ai-studio-static-online.cdn.bcebos.com/012e5e2e7a204f77ab2ad9759860978d6b19d55946dd4ee2adea6a4e9736d121)

PS：
如有需要参考项目可看：      
[基于PaddleClas2.2的从零到落地安卓部署的奥特曼分类实战](https://aistudio.baidu.com/aistudio/projectdetail/2219455)
[iFLYTEK基于PaddleClas2.2的广告分类baseline非官方](https://aistudio.baidu.com/aistudio/projectdetail/2235340)


## 2 数据集介绍

<font size="3px" color="red">本次数据集有五个可供大家选择。分别是：</font>   
1. 猫12分类
2. 垃圾40分类
3. 场景5分类
4. 食物5分类
5. 蝴蝶20分类

**数据集：都是不同类别的文件夹下放置了对应文件夹名字的类别图片**



```python
# 先导入库
from sklearn.utils import shuffle
import os
import pandas as pd
import numpy as np
from PIL import Image
import paddle
import paddle.nn as nn
import random
```


```python
# 忽略（垃圾）警告信息
# 在python中运行代码经常会遇到的情况是——代码可以正常运行但是会提示警告，有时特别讨厌。
# 那么如何来控制警告输出呢？其实很简单，python通过调用warnings模块中定义的warn()函数来发出警告。我们可以通过警告过滤器进行控制是否发出警告消息。
import warnings
warnings.filterwarnings("ignore")
```

### 2.1 解压数据集，查看数据的结构


```python
# 项目挂载的数据集先解压出来，待解压完毕，刷新后可发现左侧文件夹根目录出现五个zip
!unzip -oq /home/aistudio/data/data103736/五种图像分类数据集.zip
```

左侧可以看到如图所示五个zip    
![](https://ai-studio-static-online.cdn.bcebos.com/f8bc5b21a0ba49b4b78b6e7b18ac0341dfb14cf545b14c83b1f597b6ee8109bb)



```python
# 本项目以食物分类为例进行介绍，因为分类大多数情况下是不存在标签文件的，猫分类已经有了标签，省去了数据处理的操作
# (此处需要你根据自己的选择进行解压对应的文件)
# 解压完毕左侧出现文件夹，即为需要分类的文件
!unzip -oq /home/aistudio/食物5分类.zip
```


```python
# 查看结构，正为一个类别下有一系列对应的图片
!tree foods/
```

    5 directories, 5000 files

**五类食物图片**  
1. beef_carpaccio
2. baby_back_ribs
3. beef_tartare
4. apple_pie
5. baklava    

具体结构如下：
```
foods/
├── apple_pie
│   ├── 1005649.jpg
│   ├── 1011328.jpg
│   ├── 101251.jpg
```

### 2.2 拿到总的训练数据txt


```python
import os
# -*- coding: utf-8 -*-
# 根据官方paddleclas的提示，我们需要把图像变为两个txt文件
# train_list.txt（训练集）
# val_list.txt（验证集）
# 先把路径搞定 比如：foods/beef_carpaccio/855780.jpg ,读取到并写入txt 

# 根据左侧生成的文件夹名字来写根目录
dirpath = "foods"
# 先得到总的txt后续再进行划分，因为要划分出验证集，所以要先打乱，因为原本是有序的
def get_all_txt():
    all_list = []
    i = 0 # 标记总文件数量
    j = 0 # 标记文件类别
    for root,dirs,files in os.walk(dirpath): # 分别代表根目录、文件夹、文件
        for file in files:
            i = i + 1 
            # 文件中每行格式： 图像相对路径      图像的label_id（数字类别）（注意：中间有空格）。              
            imgpath = os.path.join(root,file)
            all_list.append(imgpath+" "+str(j)+"\n")

        j = j + 1

    allstr = ''.join(all_list)
    f = open('all_list.txt','w',encoding='utf-8')
    f.write(allstr)
    return all_list , i
all_list,all_lenth = get_all_txt()
print(all_lenth)
```

    5000


### 2.3 数据打乱


```python
# 把数据打乱
all_list = shuffle(all_list)
allstr = ''.join(all_list)
f = open('all_list.txt','w',encoding='utf-8')
f.write(allstr)
print("打乱成功，并重新写入文本")
```

    打乱成功，并重新写入文本


### 2.4 数据划分


```python
# 按照比例划分数据集 食品的数据有5000张图片，不算大数据，一般9:1即可
train_size = int(all_lenth * 0.9)
train_list = all_list[:train_size]
val_list = all_list[train_size:]

print(len(train_list))
print(len(val_list))
```

    4500
    500



```python
# 运行cell，生成训练集txt 
train_txt = ''.join(train_list)
f_train = open('train_list.txt','w',encoding='utf-8')
f_train.write(train_txt)
f_train.close()
print("train_list.txt 生成成功！")

# 运行cell，生成验证集txt
val_txt = ''.join(val_list)
f_val = open('val_list.txt','w',encoding='utf-8')
f_val.write(val_txt)
f_val.close()
print("val_list.txt 生成成功！")
```

    train_list.txt 生成成功！
    val_list.txt 生成成功！


## 3 安装paddleclas

数据集核实完搞定成功的前提下，可以准备更改原文档的参数进行实现自己的图片分类了！

这里采用paddleclas的2.2版本，好用！


```python
# 先把paddleclas安装上再说
# 安装paddleclas以及相关三方包(好像studio自带的已经够用了，无需安装了)
!git clone https://gitee.com/paddlepaddle/PaddleClas.git -b release/2.2
# 我这里安装相关包时，花了30几分钟还有错误提示，不管他即可
!pip install --upgrade -r PaddleClas/requirements.txt -i https://mirror.baidu.com/pypi/simple
```

    Cloning into 'PaddleClas'...
    remote: Enumerating objects: 538, done.[K
    remote: Counting objects: 100% (538/538), done.[K
    remote: Compressing objects: 100% (323/323), done.[K
    remote: Total 15290 (delta 346), reused 349 (delta 210), pack-reused 14752[K
    Receiving objects: 100% (15290/15290), 113.56 MiB | 9.34 MiB/s, done.
    Resolving deltas: 100% (10238/10238), done.
    Checking connectivity... done.
    Looking in indexes: https://mirror.baidu.com/pypi/simple
    Collecting prettytable (from -r PaddleClas/requirements.txt (line 1))
      Downloading https://mirror.baidu.com/pypi/packages/26/1b/42b59a4038bc0442e3a0085bc0de385658131eef8a88946333f870559b09/prettytable-2.1.0-py3-none-any.whl
    Collecting ujson (from -r PaddleClas/requirements.txt (line 2))
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/17/4e/50e8e4cf5f00b537095711c2c86ac4d7191aed2b4fffd5a19f06898f6929/ujson-4.0.2-cp37-cp37m-manylinux1_x86_64.whl (179kB)
    [K     |████████████████████████████████| 184kB 26.1MB/s eta 0:00:01
    [?25hCollecting opencv-python==4.4.0.46 (from -r PaddleClas/requirements.txt (line 3))
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/30/46/821920986c7ce5bae5518c1d490e520a9ab4cef51e3e54e35094dadf0d68/opencv-python-4.4.0.46.tar.gz (88.9MB)
    [K     |████████████████████████████████| 88.9MB 8.8MB/s eta 0:00:015    |▏                               | 614kB 11.3MB/s eta 0:00:08     |███████▋                        | 21.2MB 8.6MB/s eta 0:00:08
    [?25h  Installing build dependencies ... [?25ldone
    [?25h  Getting requirements to build wheel ... [?25ldone
    [?25h    Preparing wheel metadata ... [?25ldone
    [?25hCollecting pillow (from -r PaddleClas/requirements.txt (line 4))
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/8e/7a/b047f6f80fdb02c0cca1d3761d71e9800bcf6d4874b71c9e6548ec59e156/Pillow-8.3.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (3.0MB)
    [K     |████████████████████████████████| 3.0MB 15.9MB/s eta 0:00:01
    [?25hCollecting tqdm (from -r PaddleClas/requirements.txt (line 5))
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/0b/e8/d6f4db0886dbba2fc87b5314f2d5127acdc782e4b51e6f86972a2e45ffd6/tqdm-4.62.0-py2.py3-none-any.whl (76kB)
    [K     |████████████████████████████████| 81kB 19.0MB/s eta 0:00:01
    [?25hCollecting PyYAML (from -r PaddleClas/requirements.txt (line 6))
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/7a/a5/393c087efdc78091afa2af9f1378762f9821c9c1d7a22c5753fb5ac5f97a/PyYAML-5.4.1-cp37-cp37m-manylinux1_x86_64.whl (636kB)
    [K     |████████████████████████████████| 645kB 15.2MB/s eta 0:00:01
    [?25hRequirement already up-to-date: visualdl>=2.0.0b in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r PaddleClas/requirements.txt (line 7)) (2.2.0)
    Collecting scipy (from -r PaddleClas/requirements.txt (line 8))
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/b5/6b/8bc0b61ebf824f8c3979a31368bbe38dd247590049a994ab0ed077cb56dc/scipy-1.7.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (28.5MB)
    [K     |████████████████████████████████| 28.5MB 17.3MB/s eta 0:00:01
    [?25hCollecting scikit-learn==0.23.2 (from -r PaddleClas/requirements.txt (line 9))
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/f4/cb/64623369f348e9bfb29ff898a57ac7c91ed4921f228e9726546614d63ccb/scikit_learn-0.23.2-cp37-cp37m-manylinux1_x86_64.whl (6.8MB)
    [K     |████████████████████████████████| 6.8MB 13.8MB/s eta 0:00:01
    [?25hRequirement already up-to-date: gast==0.3.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r PaddleClas/requirements.txt (line 10)) (0.3.3)
    Requirement already satisfied, skipping upgrade: wcwidth in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from prettytable->-r PaddleClas/requirements.txt (line 1)) (0.1.7)
    Requirement already satisfied, skipping upgrade: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from prettytable->-r PaddleClas/requirements.txt (line 1)) (0.23)
    Requirement already satisfied, skipping upgrade: numpy>=1.14.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from opencv-python==4.4.0.46->-r PaddleClas/requirements.txt (line 3)) (1.20.3)
    Requirement already satisfied, skipping upgrade: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (1.21.0)
    Requirement already satisfied, skipping upgrade: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (0.8.53)
    Requirement already satisfied, skipping upgrade: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (2.2.3)
    Requirement already satisfied, skipping upgrade: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (2.22.0)
    Requirement already satisfied, skipping upgrade: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (3.14.0)
    Requirement already satisfied, skipping upgrade: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (3.8.2)
    Requirement already satisfied, skipping upgrade: six>=1.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (1.15.0)
    Requirement already satisfied, skipping upgrade: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (1.1.5)
    Requirement already satisfied, skipping upgrade: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (1.1.1)
    Requirement already satisfied, skipping upgrade: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (0.7.1.1)
    Requirement already satisfied, skipping upgrade: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (1.0.0)
    Requirement already satisfied, skipping upgrade: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn==0.23.2->-r PaddleClas/requirements.txt (line 9)) (0.14.1)
    Requirement already satisfied, skipping upgrade: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn==0.23.2->-r PaddleClas/requirements.txt (line 9)) (2.1.0)
    Requirement already satisfied, skipping upgrade: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->prettytable->-r PaddleClas/requirements.txt (line 1)) (0.6.0)
    Requirement already satisfied, skipping upgrade: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (0.10.0)
    Requirement already satisfied, skipping upgrade: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (1.4.10)
    Requirement already satisfied, skipping upgrade: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (2.0.1)
    Requirement already satisfied, skipping upgrade: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (1.3.0)
    Requirement already satisfied, skipping upgrade: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (1.3.4)
    Requirement already satisfied, skipping upgrade: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (16.7.9)
    Requirement already satisfied, skipping upgrade: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (3.9.9)
    Requirement already satisfied, skipping upgrade: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (0.18.0)
    Requirement already satisfied, skipping upgrade: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (2.4.2)
    Requirement already satisfied, skipping upgrade: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (2.8.0)
    Requirement already satisfied, skipping upgrade: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (0.10.0)
    Requirement already satisfied, skipping upgrade: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (1.1.0)
    Requirement already satisfied, skipping upgrade: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (2019.3)
    Requirement already satisfied, skipping upgrade: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (3.0.4)
    Requirement already satisfied, skipping upgrade: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (1.25.6)
    Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (2019.9.11)
    Requirement already satisfied, skipping upgrade: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (2.8)
    Requirement already satisfied, skipping upgrade: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (2.2.0)
    Requirement already satisfied, skipping upgrade: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (0.6.1)
    Requirement already satisfied, skipping upgrade: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (2.6.0)
    Requirement already satisfied, skipping upgrade: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (0.16.0)
    Requirement already satisfied, skipping upgrade: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (2.10.1)
    Requirement already satisfied, skipping upgrade: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (7.0)
    Requirement already satisfied, skipping upgrade: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (1.1.0)
    Requirement already satisfied, skipping upgrade: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (2.8.0)
    Requirement already satisfied, skipping upgrade: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->prettytable->-r PaddleClas/requirements.txt (line 1)) (7.2.0)
    Requirement already satisfied, skipping upgrade: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from kiwisolver>=1.0.1->matplotlib->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (56.2.0)
    Requirement already satisfied, skipping upgrade: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->flask>=1.1.1->visualdl>=2.0.0b->-r PaddleClas/requirements.txt (line 7)) (1.1.1)
    Building wheels for collected packages: opencv-python
      Building wheel for opencv-python (PEP 517) ... [?25ldone
    [?25h  Created wheel for opencv-python: filename=opencv_python-4.4.0.46-cp37-cp37m-linux_x86_64.whl size=12702499 sha256=aa274490c752de014c01176bcc265bc78ab38823793757df15257c70b40122ba
      Stored in directory: /home/aistudio/.cache/pip/wheels/84/ad/2c/2750e9e71f879c0807c4bbdfb84ba638eb1f9576dc211fc5bb
    Successfully built opencv-python
    [31mERROR: python-language-server 0.33.0 has requirement ujson<=1.35; platform_system != "Windows", but you'll have ujson 4.0.2 which is incompatible.[0m
    [31mERROR: python-jsonrpc-server 0.3.4 has requirement ujson<=1.35; platform_system != "Windows", but you'll have ujson 4.0.2 which is incompatible.[0m
    [31mERROR: blackhole 1.0.1 has requirement numpy<=1.19.5, but you'll have numpy 1.20.3 which is incompatible.[0m
    Installing collected packages: prettytable, ujson, opencv-python, pillow, tqdm, PyYAML, scipy, scikit-learn
      Found existing installation: prettytable 0.7.2
        Uninstalling prettytable-0.7.2:
          Successfully uninstalled prettytable-0.7.2
      Found existing installation: ujson 1.35
        Uninstalling ujson-1.35:
          Successfully uninstalled ujson-1.35
      Found existing installation: opencv-python 4.1.1.26
        Uninstalling opencv-python-4.1.1.26:
          Successfully uninstalled opencv-python-4.1.1.26
      Found existing installation: Pillow 7.1.2
        Uninstalling Pillow-7.1.2:
          Successfully uninstalled Pillow-7.1.2
      Found existing installation: tqdm 4.36.1
        Uninstalling tqdm-4.36.1:
          Successfully uninstalled tqdm-4.36.1
      Found existing installation: PyYAML 5.1.2
        Uninstalling PyYAML-5.1.2:
          Successfully uninstalled PyYAML-5.1.2
      Found existing installation: scipy 1.6.3
        Uninstalling scipy-1.6.3:
          Successfully uninstalled scipy-1.6.3
      Found existing installation: scikit-learn 0.24.2
        Uninstalling scikit-learn-0.24.2:
          Successfully uninstalled scikit-learn-0.24.2
    Successfully installed PyYAML-5.4.1 opencv-python-4.4.0.46 pillow-8.3.1 prettytable-2.1.0 scikit-learn-0.23.2 scipy-1.7.1 tqdm-4.62.0 ujson-4.0.2



```python
#因为后续paddleclas的命令需要在PaddleClas目录下，所以进入PaddleClas根目录，执行此命令
%cd PaddleClas
!ls
```

    /home/aistudio/PaddleClas
    dataset  hubconf.py   MANIFEST.in    README_ch.md  requirements.txt
    deploy	 __init__.py  paddleclas.py  README_en.md  setup.py
    docs	 LICENSE      ppcls	     README.md	   tools



```python
# 将图片移动到paddleclas下面的数据集里面
# 至于为什么现在移动，也是我的一点小技巧，防止之前移动的话，生成的txt的路径是全路径，反而需要去掉路径的一部分
!mv ../foods/ dataset/
```


```python
# 挪动文件到对应目录
!mv ../all_list.txt dataset/foods
!mv ../train_list.txt dataset/foods
!mv ../val_list.txt dataset/foods
```

### 3.1 修改配置文件
#### 3.1.1
主要是以下几点：分类数、图片总量、训练和验证的路径、图像尺寸、数据预处理、训练和预测的num_workers: 0

路径如下：
>PaddleClas/ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml

<font size="3px" color="red">（主要的参数已经进行注释，一定要过一遍）</font>

```
# global configs
Global:
  checkpoints: null
  pretrained_model: null
  output_dir: ./output/
  # 使用GPU训练
  device: gpu
  # 每几个轮次保存一次
  save_interval: 1 
  eval_during_train: True
  # 每几个轮次验证一次
  eval_interval: 1 
  # 训练轮次
  epochs: 20 
  print_batch_step: 1
  use_visualdl: True #开启可视化（目前平台不可用）
  # used for static mode and model export
  # 图像大小
  image_shape: [3, 224, 224] 
  save_inference_dir: ./inference
  # training model under @to_static
  to_static: False

# model architecture
Arch:
  # 采用的网络
  name: ResNet50
  # 类别数 多了个0类 0-5 0无用 
  class_num: 6 
 
# loss function config for traing/eval process
Loss:
  Train:

    - CELoss: 
        weight: 1.0
  Eval:
    - CELoss:
        weight: 1.0


Optimizer:
  name: Momentum
  momentum: 0.9
  lr:
    name: Piecewise
    learning_rate: 0.015
    decay_epochs: [30, 60, 90]
    values: [0.1, 0.01, 0.001, 0.0001]
  regularizer:
    name: 'L2'
    coeff: 0.0005


# data loader for train and eval
DataLoader:
  Train:
    dataset:
      name: ImageNetDataset
      # 根路径
      image_root: ./dataset/
      # 前面自己生产得到的训练集文本路径
      cls_label_path: ./dataset/foods/train_list.txt
      # 数据预处理
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - ResizeImage:
            resize_short: 256
        - CropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''

    sampler:
      name: DistributedBatchSampler
      batch_size: 128
      drop_last: False
      shuffle: True
    loader:
      num_workers: 0
      use_shared_memory: True

  Eval:
    dataset: 
      name: ImageNetDataset
      # 根路径
      image_root: ./dataset/
      # 前面自己生产得到的验证集文本路径
      cls_label_path: ./dataset/foods/val_list.txt
      # 数据预处理
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - ResizeImage:
            resize_short: 256
        - CropImage:
            size: 224
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
    sampler:
      name: DistributedBatchSampler
      batch_size: 128
      drop_last: False
      shuffle: True
    loader:
      num_workers: 0
      use_shared_memory: True

Infer:
  infer_imgs: ./dataset/foods/beef_carpaccio/855780.jpg
  batch_size: 10
  transforms:
    - DecodeImage:
        to_rgb: True
        channel_first: False
    - ResizeImage:
        resize_short: 256
    - CropImage:
        size: 224
    - NormalizeImage:
        scale: 1.0/255.0
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
        order: ''
    - ToCHWImage:
  PostProcess:
    name: Topk
    # 输出的可能性最高的前topk个
    topk: 5
    # 标签文件 需要自己新建文件
    class_id_map_file: ./dataset/label_list.txt

Metric:
  Train:
    - TopkAcc:
        topk: [1, 5]
  Eval:
    - TopkAcc:
        topk: [1, 5]
```
#### 3.1.2 标签文件
这个是在预测时生成对照的依据，在上个文件有提到这个
```
# 标签文件 需要自己新建文件
    class_id_map_file: dataset/label_list.txt
```
按照对应的进行编写：   

![](https://ai-studio-static-online.cdn.bcebos.com/0e40a0afaa824ba9b70778aa7931a3baf2a421bcb81b4b0f83632da4e4ddc0ef)  

如食品分类(要对照之前的txt的类别确认无误) 
```
1 beef_carpaccio
2 baby_back_ribs
3 beef_tartare
4 apple_pie
5 baklava
```
![](https://ai-studio-static-online.cdn.bcebos.com/6b3a4a244ed34517bcf4be5fcc629ab10d081eb0af7c4532a795583d19939a82)




![](https://ai-studio-static-online.cdn.bcebos.com/8ee69dbad8bf4e0f9c743645a4438cc035446d052d4b409a88c1cc25b58dcf83)

### 3.2 开始训练


```python
# 提示，运行过程中可能存在坏图的情况，但是不用担心，训练过程不受影响。
# 仅供参考，我只跑了五轮，准确率很低
!python3 tools/train.py \
    -c ./ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml
```

    [2021/08/12 15:01:58] root INFO: Already save model in ./output/ShuffleNetV2_x0_25/latest

### 3.3 预测一张


```python
# 更换为你训练的网络，需要预测的文件，上面训练所得到的的最优模型文件
# 我这里是不严谨的，直接使用训练集的图片进行验证，大家可以去百度搜一些相关的图片传上来，进行预测
!python3 tools/infer.py \
    -c ./ppcls/configs/quick_start/new_user/ShuffleNetV2_x0_25.yaml \
    -o Infer.infer_imgs=dataset/foods/baby_back_ribs/319516.jpg \
    -o Global.pretrained_model=output/ShuffleNetV2_x0_25/best_model
```

    /home/aistudio/PaddleClas/ppcls/arch/backbone/model_zoo/vision_transformer.py:15: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Callable
    [2021/08/12 15:05:43] root INFO: 
    ===========================================================
    ==        PaddleClas is powered by PaddlePaddle !        ==
    ===========================================================
    ==                                                       ==
    ==   For more info please go to the following website.   ==
    ==                                                       ==
    ==       https://github.com/PaddlePaddle/PaddleClas      ==
    ===========================================================
    
    [2021/08/12 15:05:43] root INFO: Arch : 
    [2021/08/12 15:05:43] root INFO:     class_num : 6
    [2021/08/12 15:05:43] root INFO:     name : ShuffleNetV2_x0_25
    [2021/08/12 15:05:43] root INFO: DataLoader : 
    [2021/08/12 15:05:43] root INFO:     Eval : 
    [2021/08/12 15:05:43] root INFO:         dataset : 
    [2021/08/12 15:05:43] root INFO:             cls_label_path : ./dataset/foods/val_list.txt
    [2021/08/12 15:05:43] root INFO:             image_root : ./dataset/
    [2021/08/12 15:05:43] root INFO:             name : ImageNetDataset
    [2021/08/12 15:05:43] root INFO:             transform_ops : 
    [2021/08/12 15:05:43] root INFO:                 DecodeImage : 
    [2021/08/12 15:05:43] root INFO:                     channel_first : False
    [2021/08/12 15:05:43] root INFO:                     to_rgb : True
    [2021/08/12 15:05:43] root INFO:                 ResizeImage : 
    [2021/08/12 15:05:43] root INFO:                     resize_short : 256
    [2021/08/12 15:05:43] root INFO:                 CropImage : 
    [2021/08/12 15:05:43] root INFO:                     size : 224
    [2021/08/12 15:05:43] root INFO:                 NormalizeImage : 
    [2021/08/12 15:05:43] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2021/08/12 15:05:43] root INFO:                     order : 
    [2021/08/12 15:05:43] root INFO:                     scale : 1.0/255.0
    [2021/08/12 15:05:43] root INFO:                     std : [0.229, 0.224, 0.225]
    [2021/08/12 15:05:43] root INFO:         loader : 
    [2021/08/12 15:05:43] root INFO:             num_workers : 0
    [2021/08/12 15:05:43] root INFO:             use_shared_memory : True
    [2021/08/12 15:05:43] root INFO:         sampler : 
    [2021/08/12 15:05:43] root INFO:             batch_size : 128
    [2021/08/12 15:05:43] root INFO:             drop_last : False
    [2021/08/12 15:05:43] root INFO:             name : DistributedBatchSampler
    [2021/08/12 15:05:43] root INFO:             shuffle : True
    [2021/08/12 15:05:43] root INFO:     Train : 
    [2021/08/12 15:05:43] root INFO:         dataset : 
    [2021/08/12 15:05:43] root INFO:             cls_label_path : ./dataset/foods/train_list.txt
    [2021/08/12 15:05:43] root INFO:             image_root : ./dataset/
    [2021/08/12 15:05:43] root INFO:             name : ImageNetDataset
    [2021/08/12 15:05:43] root INFO:             transform_ops : 
    [2021/08/12 15:05:43] root INFO:                 DecodeImage : 
    [2021/08/12 15:05:43] root INFO:                     channel_first : False
    [2021/08/12 15:05:43] root INFO:                     to_rgb : True
    [2021/08/12 15:05:43] root INFO:                 RandCropImage : 
    [2021/08/12 15:05:43] root INFO:                     size : 224
    [2021/08/12 15:05:43] root INFO:                 RandFlipImage : 
    [2021/08/12 15:05:43] root INFO:                     flip_code : 1
    [2021/08/12 15:05:43] root INFO:                 NormalizeImage : 
    [2021/08/12 15:05:43] root INFO:                     mean : [0.485, 0.456, 0.406]
    [2021/08/12 15:05:43] root INFO:                     order : 
    [2021/08/12 15:05:43] root INFO:                     scale : 1.0/255.0
    [2021/08/12 15:05:43] root INFO:                     std : [0.229, 0.224, 0.225]
    [2021/08/12 15:05:43] root INFO:         loader : 
    [2021/08/12 15:05:43] root INFO:             num_workers : 0
    [2021/08/12 15:05:43] root INFO:             use_shared_memory : True
    [2021/08/12 15:05:43] root INFO:         sampler : 
    [2021/08/12 15:05:43] root INFO:             batch_size : 128
    [2021/08/12 15:05:43] root INFO:             drop_last : False
    [2021/08/12 15:05:43] root INFO:             name : DistributedBatchSampler
    [2021/08/12 15:05:43] root INFO:             shuffle : True
    [2021/08/12 15:05:43] root INFO: Global : 
    [2021/08/12 15:05:43] root INFO:     checkpoints : None
    [2021/08/12 15:05:43] root INFO:     device : gpu
    [2021/08/12 15:05:43] root INFO:     epochs : 20
    [2021/08/12 15:05:43] root INFO:     eval_during_train : True
    [2021/08/12 15:05:43] root INFO:     eval_interval : 1
    [2021/08/12 15:05:43] root INFO:     image_shape : [3, 224, 224]
    [2021/08/12 15:05:43] root INFO:     output_dir : ./output/
    [2021/08/12 15:05:43] root INFO:     pretrained_model : output/ShuffleNetV2_x0_25/best_model
    [2021/08/12 15:05:43] root INFO:     print_batch_step : 1
    [2021/08/12 15:05:43] root INFO:     save_inference_dir : ./inference
    [2021/08/12 15:05:43] root INFO:     save_interval : 1
    [2021/08/12 15:05:43] root INFO:     use_visualdl : True
    [2021/08/12 15:05:43] root INFO: Infer : 
    [2021/08/12 15:05:43] root INFO:     PostProcess : 
    [2021/08/12 15:05:43] root INFO:         name : Topk
    [2021/08/12 15:05:43] root INFO:         topk : 5
    [2021/08/12 15:05:43] root INFO:     batch_size : 10
    [2021/08/12 15:05:43] root INFO:     infer_imgs : dataset/foods/baby_back_ribs/319516.jpg
    [2021/08/12 15:05:43] root INFO:     transforms : 
    [2021/08/12 15:05:43] root INFO:         DecodeImage : 
    [2021/08/12 15:05:43] root INFO:             channel_first : False
    [2021/08/12 15:05:43] root INFO:             to_rgb : True
    [2021/08/12 15:05:43] root INFO:         ResizeImage : 
    [2021/08/12 15:05:43] root INFO:             resize_short : 256
    [2021/08/12 15:05:43] root INFO:         CropImage : 
    [2021/08/12 15:05:43] root INFO:             size : 224
    [2021/08/12 15:05:43] root INFO:         NormalizeImage : 
    [2021/08/12 15:05:43] root INFO:             mean : [0.485, 0.456, 0.406]
    [2021/08/12 15:05:43] root INFO:             order : 
    [2021/08/12 15:05:43] root INFO:             scale : 1.0/255.0
    [2021/08/12 15:05:43] root INFO:             std : [0.229, 0.224, 0.225]
    [2021/08/12 15:05:43] root INFO:         ToCHWImage : None
    [2021/08/12 15:05:43] root INFO: Loss : 
    [2021/08/12 15:05:43] root INFO:     Eval : 
    [2021/08/12 15:05:43] root INFO:         CELoss : 
    [2021/08/12 15:05:43] root INFO:             weight : 1.0
    [2021/08/12 15:05:43] root INFO:     Train : 
    [2021/08/12 15:05:43] root INFO:         CELoss : 
    [2021/08/12 15:05:43] root INFO:             weight : 1.0
    [2021/08/12 15:05:43] root INFO: Metric : 
    [2021/08/12 15:05:43] root INFO:     Eval : 
    [2021/08/12 15:05:43] root INFO:         TopkAcc : 
    [2021/08/12 15:05:43] root INFO:             topk : [1, 5]
    [2021/08/12 15:05:43] root INFO:     Train : 
    [2021/08/12 15:05:43] root INFO:         TopkAcc : 
    [2021/08/12 15:05:43] root INFO:             topk : [1, 5]
    [2021/08/12 15:05:43] root INFO: Optimizer : 
    [2021/08/12 15:05:43] root INFO:     lr : 
    [2021/08/12 15:05:43] root INFO:         learning_rate : 0.015
    [2021/08/12 15:05:43] root INFO:         name : Cosine
    [2021/08/12 15:05:43] root INFO:         warmup_epoch : 5
    [2021/08/12 15:05:43] root INFO:     momentum : 0.9
    [2021/08/12 15:05:43] root INFO:     name : Momentum
    [2021/08/12 15:05:43] root INFO:     regularizer : 
    [2021/08/12 15:05:43] root INFO:         coeff : 0.0005
    [2021/08/12 15:05:43] root INFO:         name : L2
    W0812 15:05:43.023571  8050 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0812 15:05:43.028319  8050 device_context.cc:422] device: 0, cuDNN Version: 7.6.
    [2021/08/12 15:05:48] root INFO: train with paddle 2.1.2 and device CUDAPlace(0)
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:125: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if data.dtype == np.object:
    [{'class_ids': [2, 4, 3, 1, 5], 'scores': [0.67784, 0.16818, 0.09998, 0.04119, 0.0128], 'file_name': 'dataset/foods/baby_back_ribs/319516.jpg', 'label_names': []}]


运行完成，最后几行会得到结果如下形式：
```
[{'class_ids': [5, 1, 3, 4, 2],
'scores': [0.48433, 0.26765, 0.13903, 0.05609, 0.05162],
'file_name': 'dataset/foods/baby_back_ribs/319516.jpg', 
'label_names': ['baklava', 'beef_carpaccio', 'beef_tartare', 'apple_pie', 'baby_back_ribs']}]
```
可以发现，预测结果不对，准确率很低，但整体的项目流程你已经掌握了！    
训练轮数还有很大提升空间，自行变动参数直到预测正确为止~    

<font size="3px" color="red">恭喜你学会了paddleclas图像分类！</font>

## 最后（！很重要！）    
1. 生成版本     
![](https://ai-studio-static-online.cdn.bcebos.com/adda3ac3b96c403daa68f9f5ec9fd807930d1404f5834317a4f3d7e662243ebc)   
2. 有能力者导出markdown发表到github上      
![](https://ai-studio-static-online.cdn.bcebos.com/a1f8fdad7659463381d8857e9fd63e84648591fc1a264f1a8241029fb825fe78)      
3. 公开项目   
![](https://ai-studio-static-online.cdn.bcebos.com/bc8af42d84914b7a8e882e3b88551aa7a5943cf7351747b2b6df3310be53eb05)    
4. <font color="red">祝大家顺利结业~</font>
