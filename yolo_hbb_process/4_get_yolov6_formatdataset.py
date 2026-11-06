import os
import shutil

train_img_dir=r'./out/images/train/'
train_label_dir=r'./out/labels/train/'
val_img_dir=r'./out/images/val/'
val_label_dir=r'./out/labels/val/'

if not os.path.exists(train_img_dir):
    os.makedirs(train_img_dir)

if not os.path.exists(train_label_dir):
    os.makedirs(train_label_dir)

if not os.path.exists(val_img_dir):
    os.makedirs(val_img_dir)

if not os.path.exists(val_label_dir):
    os.makedirs(val_label_dir)


with open('train.txt','r') as ft:
    lines=ft.readlines()
    for line in lines:
        line=line.strip()
        shutil.copy('./images/'+line,train_img_dir+line)

        label_name=line[:-4]
        label_line=label_name+'.txt'
        shutil.copy('./labels/'+label_line,train_label_dir+label_line)


with open('val.txt','r') as fv:
    lines = fv.readlines()
    for line in lines:
        line = line.strip()
        shutil.copy('./images/' + line, val_img_dir + line)

        label_name=line[:-4]
        label_line=label_name+'.txt'
        shutil.copy('./labels/'+label_line,val_label_dir+label_line)