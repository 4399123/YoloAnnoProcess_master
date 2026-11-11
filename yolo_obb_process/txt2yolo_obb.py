# coding:utf-8
"""
ICDAR / TXT 标签 -> Ultralytics YOLO OBB 标签转换工具

主要功能:
    1. 将 labels 文件夹下的 txt 文本标注转换为 YOLO 官方 OBB 格式:
       <class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
    2. 类别 id 统一设置为 0 (支持通过 --cls-id 自定义);
    3. 去除尾部转录文本内容, 仅保留 id 和归一化坐标;
    4. 自动联动 images 文件夹下对应的图片尺寸 (宽 W, 高 H) 进行归一化 [0, 1];
    5. 自动匹配图片与标签 (如 gt_img_1.txt 对应 img_1.jpg, 输出 img_1.txt, 符合 YOLO 规范).

参考格式:
    0 0.174037 0.053367 0.368342 0.053367 0.368342 0.178794 0.174037 0.178794

用法示例:
    python txt2yolo_obb.py
    python txt2yolo_obb.py --output labels_yolo
    python txt2yolo_obb.py --output labels
    python txt2yolo_obb.py --images images --labels labels --output labels_yolo --skip-difficult
"""

import os
import sys
import glob
import time
import argparse
from PIL import Image
import numpy as np

# 支持的图像后缀
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def clip01(v):
    """截断数值到 [0.0, 1.0] 范围内"""
    return min(max(float(v), 0.0), 1.0)


def get_image_size(img_path):
    """
    快速获取图像尺寸 (宽, 高).
    优先使用 PIL 仅读取文件头 (速度极快, <1ms/张); 遇到异常则使用 cv2.imdecode 兜底.
    """
    try:
        with Image.open(img_path) as im:
            return im.size  # (width, height)
    except Exception:
        pass

    try:
        import cv2
        data = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    except Exception:
        pass

    return None, None


def find_matching_image(label_filename, images_dir):
    """
    根据标签文件名在 images_dir 中寻找对应的图像路径:
    1. 若标签为 gt_img_1.txt, 尝试 img_1.* 以及 gt_img_1.*
    2. 遍历常见图像扩展名
    """
    stem = os.path.splitext(label_filename)[0]
    candidate_stems = []

    # 优先去除 gt_ 前缀匹配图像 (如 gt_img_1 -> img_1)
    if stem.startswith("gt_"):
        candidate_stems.append(stem[3:])
    candidate_stems.append(stem)

    for cand in candidate_stems:
        for ext in IMG_EXTS:
            for e in (ext, ext.upper()):
                img_path = os.path.join(images_dir, cand + e)
                if os.path.isfile(img_path):
                    return img_path, cand

    return None, stem


def parse_line(line):
    """
    解析单行标注.
    ICDAR 格式通常为: x1,y1,x2,y2,x3,y3,x4,y4,transcription
    支持逗号分隔, 也兼容空格/制表符分隔.
    返回: (coords: list of 8 floats, text: str) 或 None
    """
    line = line.strip()
    if not line:
        return None

    # 优先按逗号拆分
    if "," in line:
        parts = line.split(",")
    else:
        parts = line.split()

    if len(parts) < 8:
        return None

    try:
        coords = [float(p.strip()) for p in parts[:8]]
    except ValueError:
        return None

    text = ",".join(parts[8:]).strip() if len(parts) > 8 else ""
    return coords, text


def convert_dataset(labels_dir, images_dir, output_dir, cls_id=0,
                    keep_prefix=False, skip_difficult=False, precision=6):
    """执行批量转换"""
    if not os.path.isdir(labels_dir):
        print(f"[错误] 标签目录不存在: {labels_dir}")
        return False
    if not os.path.isdir(images_dir):
        print(f"[错误] 图片目录不存在: {images_dir}")
        return False

    os.makedirs(output_dir, exist_ok=True)

    # 收集待转换的 txt 标签文件
    all_txts = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    if not all_txts:
        print(f"[警告] {labels_dir} 中未找到任何 .txt 文件")
        return False

    # 若输入目录与输出目录相同, 避免重新处理已生成的同名输出文件
    same_dir = os.path.abspath(labels_dir) == os.path.abspath(output_dir)

    print("=" * 60)
    print("开始转换标签为 YOLO OBB 官方格式:")
    print(f"  - 标签目录: {os.path.abspath(labels_dir)} ({len(all_txts)} 个 txt 文件)")
    print(f"  - 图像目录: {os.path.abspath(images_dir)}")
    print(f"  - 输出目录: {os.path.abspath(output_dir)}")
    print(f"  - 目标类别 ID: {cls_id}")
    print(f"  - 过滤 ### 难例: {'是' if skip_difficult else '否 (全部保留为类别 0)'}")
    print(f"  - 浮点坐标精度: {precision} 位小数")
    print("=" * 60)

    start_time = time.time()
    n_converted_files = 0
    n_total_boxes = 0
    n_skipped_boxes = 0
    n_missing_images = 0

    format_str = f"{{:.{precision}f}}"

    for txt_path in all_txts:
        fname = os.path.basename(txt_path)
        img_path, matched_stem = find_matching_image(fname, images_dir)

        if not img_path:
            print(f"[警告] 未找到对应原图, 跳过标签: {fname}")
            n_missing_images += 1
            continue

        w, h = get_image_size(img_path)
        if w is None or h is None or w <= 0 or h <= 0:
            print(f"[警告] 无法读取图像尺寸: {img_path}, 跳过标签: {fname}")
            continue

        # 确定输出文件名: 默认采用与原图同名 (img_1.txt), 便于 YOLO 标准读取
        if keep_prefix:
            out_fname = fname
        else:
            out_fname = matched_stem + ".txt"

        out_path = os.path.join(output_dir, out_fname)

        # 尝试使用 utf-8-sig (自动去除可能存在的 BOM), 失败则降级到 gbk/latin1
        lines = []
        for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
            try:
                with open(txt_path, "r", encoding=enc) as f:
                    lines = f.readlines()
                break
            except (UnicodeDecodeError, LookupError):
                continue

        converted_lines = []
        for line in lines:
            parsed = parse_line(line)
            if not parsed:
                continue

            coords, text = parsed

            # 若开启了过滤 ### 难例
            if skip_difficult and "###" in text:
                n_skipped_boxes += 1
                continue

            # 坐标归一化并截断到 [0, 1]
            # 8个坐标顺序: x1, y1, x2, y2, x3, y3, x4, y4
            norm_coords = []
            for i in range(0, 8, 2):
                nx = clip01(coords[i] / w)
                ny = clip01(coords[i + 1] / h)
                norm_coords.extend([nx, ny])

            # 拼装为 YOLO OBB 官方格式: class_id x1 y1 x2 y2 x3 y3 x4 y4
            coord_strs = [format_str.format(c) for c in norm_coords]
            line_str = f"{cls_id} " + " ".join(coord_strs)
            converted_lines.append(line_str)
            n_total_boxes += 1

        # 写入转换结果
        with open(out_path, "w", encoding="utf-8") as f:
            for l in converted_lines:
                f.write(l + "\n")

        n_converted_files += 1

    elapsed = time.time() - start_time
    print("-" * 60)
    print("转换完成!")
    print(f"  - 成功转换文件数: {n_converted_files}")
    print(f"  - 转换目标框总数: {n_total_boxes}")
    if skip_difficult:
        print(f"  - 过滤难例框数 (###): {n_skipped_boxes}")
    if n_missing_images > 0:
        print(f"  - 缺少对应图像的文件数: {n_missing_images}")
    print(f"  - 耗时: {elapsed:.2f} 秒")
    print(f"  - 输出路径: {os.path.abspath(output_dir)}")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="将 labels 下的 txt 标注转为 YOLO 官方 OBB 格式")
    parser.add_argument("--labels", default="labels", type=str, help="输入标签 txt 文件夹路径")
    parser.add_argument("--images", default="images", type=str, help="输入图像文件夹路径")
    parser.add_argument("--output", default="labels_yolo", type=str,
                        help="输出标签文件夹路径 (例如 labels_yolo 或 labels)")
    parser.add_argument("--cls-id", default=0, type=int, help="目标类别 ID, 默认 0")
    parser.add_argument("--keep-prefix", action="store_true",
                        help="保留原 txt 的文件名 (如保留 gt_ 前缀), 默认会自动去除 gt_ 与图像同名")
    parser.add_argument("--skip-difficult", action="store_true",
                        help="过滤掉包含 '###' 的难例标注框 (默认保留并设为类别 0)")
    parser.add_argument("--precision", default=6, type=int, help="归一化坐标浮点精度 (默认保留 6 位小数)")

    args = parser.parse_args()

    convert_dataset(
        labels_dir=args.labels,
        images_dir=args.images,
        output_dir=args.output,
        cls_id=args.cls_id,
        keep_prefix=args.keep_prefix,
        skip_difficult=args.skip_difficult,
        precision=args.precision
    )


if __name__ == "__main__":
    main()
