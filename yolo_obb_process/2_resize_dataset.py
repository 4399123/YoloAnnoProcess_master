# coding:utf-8
"""
图像 + YOLO OBB 标签 联动 resize (直接拉伸, 不保持宽高比, 不加黑边)

输入:
    images/xxx.png   原图
    labels/xxx.txt   标签: class_index x1 y1 x2 y2 x3 y3 x4 y4 (归一化坐标)
输出 (原 images / labels 不做任何修改):
    imagesX/xxx.png  resize 后的图像 (cv2.INTER_LINEAR 双线性插值)
    labelsX/xxx.txt  换算后的标签, class_index 保持不变

说明:
    直接 resize 时 x 方向缩放 sx = W/W0, y 方向缩放 sy = H/H0,
    像素坐标 (px, py) -> (px*sx, py*sy), 再除以新尺寸 (W, H) 归一化。
    因此归一化坐标在数学上保持不变, 输出值与原值一致 (仅有浮点舍入差异)。

跳过规则:
    - txt 为空 (无目标) 的样本: 跳过坐标换算, 图像照常 resize, 输出空 txt
    - 没有对应 txt 的图像: 跳过并提示

用法示例:
    python 2_resize_dataset.py                    # 默认 640x640
    python 2_resize_dataset.py --W 1024 --H 768 --images images --labels labels --out_images imagesX --out_labels labelsX
"""

import os
import argparse

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

parser = argparse.ArgumentParser()
parser.add_argument('--W', default=800, type=int, help='目标宽度')
parser.add_argument('--H', default=800, type=int, help='目标高度')
parser.add_argument('--images', default=r'images', type=str, help='原图文件夹')
parser.add_argument('--labels', default=r'labels', type=str, help='原标签文件夹')
parser.add_argument('--out_images', default=r'imagesX', type=str, help='输出图像文件夹')
parser.add_argument('--out_labels', default=r'labelsX', type=str, help='输出标签文件夹')
opt = parser.parse_args()


def imread(path):
    # 兼容中文路径; IMREAD_UNCHANGED 保留原始通道数/位深
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def imwrite(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok:
        buf.tofile(path)
    return ok


def convert_label(label_path, src_w, src_h, dst_w, dst_h):
    """读取标签并换算坐标, 返回新的行列表; 格式错误的行打印警告并跳过。"""
    sx, sy = dst_w / src_w, dst_h / src_h
    lines = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 9:
                tqdm.write('[警告] {} 第 {} 行字段数为 {} (应为 9), 跳过'.format(label_path, ln, len(parts)))
                continue
            cls_id = parts[0]  # 类别 id 原样保留
            coords = np.array(parts[1:], dtype=np.float64).reshape(4, 2)
            # 归一化 -> 原图像素 -> 新图像素 -> 新图归一化
            px = coords * [src_w, src_h]
            px = px * [sx, sy]
            norm = np.clip(px / [dst_w, dst_h], 0.0, 1.0)
            lines.append(cls_id + ' ' + ' '.join('{:.6f}'.format(v) for v in norm.ravel()))
    return lines


def main():
    dst_w, dst_h = opt.W, opt.H
    if dst_w <= 0 or dst_h <= 0:
        raise SystemExit('目标分辨率必须为正整数')

    os.makedirs(opt.out_images, exist_ok=True)
    os.makedirs(opt.out_labels, exist_ok=True)

    n_done = n_empty = n_nolabel = n_bad = 0
    img_names = [name for name in sorted(os.listdir(opt.images)) if name.lower().endswith(IMG_EXTS)]
    pbar = tqdm(img_names, desc='Resize数据集')
    for name in pbar:
        stem = os.path.splitext(name)[0]
        label_path = os.path.join(opt.labels, stem + '.txt')

        if not os.path.isfile(label_path):
            tqdm.write('[提示] 缺少标签, 跳过: {}'.format(name))
            n_nolabel += 1
            continue
        img = imread(os.path.join(opt.images, name))
        if img is None:
            tqdm.write('[警告] 无法读取图片, 跳过: {}'.format(name))
            n_bad += 1
            continue
        src_h, src_w = img.shape[:2]

        # 空标签 (背景图): 跳过坐标换算, 但图像照常 resize, 并输出空 txt 保持一一对应
        with open(label_path, 'r', encoding='utf-8') as f:
            is_empty = not f.read().strip()
        lines = [] if is_empty else convert_label(label_path, src_w, src_h, dst_w, dst_h)
        if not lines:
            tqdm.write('[提示] 标签为空, 仅 resize 图像: {}'.format(name))
            n_empty += 1

        resized = cv2.resize(img, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
        if not imwrite(os.path.join(opt.out_images, name), resized):
            tqdm.write('[警告] 图片保存失败: {}'.format(name))
            n_bad += 1
            continue
        with open(os.path.join(opt.out_labels, stem + '.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n' if lines else '')
        n_done += 1

    print('目标分辨率: {}x{}'.format(dst_w, dst_h))
    print('完成: {} 张 (其中空标签 {}); 缺标签跳过 {}, 读写失败 {}'.format(
        n_done, n_empty, n_nolabel, n_bad))
    print('输出: {} | {}'.format(os.path.abspath(opt.out_images), os.path.abspath(opt.out_labels)))


if __name__ == '__main__':
    main()
