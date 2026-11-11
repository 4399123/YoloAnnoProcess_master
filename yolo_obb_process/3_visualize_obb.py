"""
YOLO OBB 标签可视化工具: 把 labels/*.txt 中的旋转框画回 images/ 中的原图, 用于检查标注是否准确。

标签格式 (Ultralytics OBB): class_index x1 y1 x2 y2 x3 y3 x4 y4 (归一化坐标)
结果保存到 suoying_output/ (文件名与原图相同)。

用法示例:
    python visualize_obb.py
    python visualize_obb.py --images images --labels labels --output suoying_output --names BS
"""

import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

# class_index -> 类别名, 需与转换时的 CLASSES 顺序一致 (Other 已被过滤)
NAMES = ["wenzi"]
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# 每个类别一种颜色 (BGR)
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
          (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 255)]


def imread(path):
    # 用 imdecode 读取, 兼容中文路径
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def draw_labels(img, label_path, names):
    h, w = img.shape[:2]
    thickness = max(1, round(min(h, w) / 400))
    count = 0
    with open(label_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 9:
                tqdm.write(f"[警告] {label_path} 第 {ln} 行字段数为 {len(parts)} (应为 9), 跳过")
                continue
            cls_id = int(parts[0])
            coords = np.array(parts[1:], dtype=np.float64).reshape(4, 2)
            if (coords < 0).any() or (coords > 1).any():
                tqdm.write(f"[警告] {label_path} 第 {ln} 行坐标超出 [0,1]")
            pts = np.round(coords * [w, h]).astype(np.int32)

            color = COLORS[cls_id % len(COLORS)]
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
            # 标出第一个点 (x1, y1), 方便检查角点顺序
            cv2.circle(img, tuple(int(v) for v in pts[0]), thickness + 2, color, -1)

            name = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
            x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
            cv2.putText(img, name, (x, max(y - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5 * thickness, color, thickness, cv2.LINE_AA)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="将 YOLO OBB 标签画到原图上进行核对")
    parser.add_argument("--images", default="imagesX", help="原图文件夹")
    parser.add_argument("--labels", default="labelsX", help="标签 txt 文件夹")
    parser.add_argument("--output", default="suoying_output", help="结果图输出文件夹")
    parser.add_argument("--names", nargs="+", default=NAMES, help="类别名, 顺序对应 class_index")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    imgs = sorted(f for f in os.listdir(args.images) if f.lower().endswith(IMG_EXTS))

    n_img = n_obj = n_nolabel = 0
    pbar = tqdm(imgs, desc="可视化标签")
    for name in pbar:
        img = imread(os.path.join(args.images, name))
        if img is None:
            tqdm.write(f"[警告] 无法读取图片: {name}")
            continue
        label_path = os.path.join(args.labels, os.path.splitext(name)[0] + ".txt")
        if os.path.isfile(label_path):
            n_obj += draw_labels(img, label_path, args.names)
        else:
            tqdm.write(f"[提示] 无对应标签: {name}")
            n_nolabel += 1
        imwrite(os.path.join(args.output, name), img)
        n_img += 1

    print(f"完成: {n_img} 张图片, {n_obj} 个框, 缺少标签 {n_nolabel} 张")
    print(f"输出: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
