# coding:utf-8
"""
按比例划分 YOLO OBB 数据集 (参考 1_split_train_val.py 的划分方式)

输入:
    images/xxx.png   原图
    labels/xxx.txt   标签 (与图像同名)
输出:
    dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

用法示例:
    python 2_split_dataset.py
    python 2_split_dataset.py --images images --labels labels --output dataset --train_percent 0.9 --move
"""

import os
import random
import shutil
import argparse

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

parser = argparse.ArgumentParser()
# 原图文件夹
parser.add_argument('--images', default=r'imagesX', type=str, help='input image path')
# 标签文件夹 (txt 与图像同名)
parser.add_argument('--labels', default=r'labelsX', type=str, help='input label path')
# 输出数据集根目录, 自动创建
parser.add_argument('--output', default=r'dataset', type=str, help='output dataset path')
parser.add_argument('--train_percent', default=0.9, type=float, help='train ratio')
parser.add_argument('--move', action='store_true', help='move files instead of copy')
opt = parser.parse_args()

random.seed(42)
train_percent = opt.train_percent

# 只保留图像和标签都存在的样本, 保证一一对应
pairs = []
for name in sorted(os.listdir(opt.images)):
    if not name.lower().endswith(IMG_EXTS):
        continue
    stem = os.path.splitext(name)[0]
    label = stem + '.txt'
    if not os.path.isfile(os.path.join(opt.labels, label)):
        print('[警告] 缺少标签, 跳过: {}'.format(name))
        continue
    pairs.append((name, label))

num = len(pairs)
tr = int(num * train_percent)
index = list(range(num))
random.shuffle(index)
train = set(index[:tr])
print('train num:{}'.format(tr))
print('val num:{}'.format(num - tr))

for split in ('train', 'val'):
    os.makedirs(os.path.join(opt.output, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(opt.output, 'labels', split), exist_ok=True)

op = shutil.move if opt.move else shutil.copy2
for i, (img, label) in enumerate(pairs):
    split = 'train' if i in train else 'val'
    op(os.path.join(opt.images, img), os.path.join(opt.output, 'images', split, img))
    op(os.path.join(opt.labels, label), os.path.join(opt.output, 'labels', split, label))

print('输出: {}'.format(os.path.abspath(opt.output)))
