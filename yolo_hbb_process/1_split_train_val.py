# coding:utf-8

import os
import random
import argparse

parser = argparse.ArgumentParser()
#xml文件的地址，根据自己的数据进行修改 xml一般存放在Annotations下
parser.add_argument('--xml_path', default=r'Annotations', type=str, help='input xml label path')
#数据集的划分，地址选择自己数据下的ImageSets,该文件夹自动创建
parser.add_argument('--txt_path', default=r'ImageSets', type=str, help='output txt label path')
opt = parser.parse_args()

random.seed(42)
trainval_percent = 1.0
train_percent = 0.9
xmlfilepath = opt.xml_path
txtsavepath = opt.txt_path
total_xml = os.listdir(xmlfilepath)
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

num = len(total_xml)
list_index = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list_index, tv)
random.shuffle(trainval)
train = random.sample(trainval, tr)
print('train num:{}'.format(tr))
print('val num:{}'.format(tv-tr))

file_train = open(txtsavepath + '/train.txt', 'w')
file_val = open(txtsavepath + '/val.txt', 'w')

for i in trainval:
    name = total_xml[i][:-4] + '\n'
    if i in train:
        file_train.write(name)
    else:
        file_val.write(name)

file_train.close()
file_val.close()
