# -*- coding: utf-8 -*-
"""
MSRA-TD500 标注转换为 YOLO26 OBB 官方旋转框检测格式脚本。

【MSRA-TD500 原始标注格式 (.gt 或 .txt)】:
    每行表示一个文本行:
    index difficulty_flag x y w h theta
    - index: 目标索引编号
    - difficulty_flag: 难度标记 (0: 正常, 1: 模糊/遮挡/困难文本)
    - x, y: 旋转矩形未旋转前的外接矩形左上角坐标 (像素)
    - w, h: 旋转矩形的宽与高 (像素)
    - theta: 旋转弧度 (以矩形中心为原点旋转, 取值通常在 [-pi/2, pi/2] 范围)

【YOLO26 OBB 官方训练格式】:
    每行一个目标，坐标归一化到 0~1:
    class_id x1 y1 x2 y2 x3 y3 x4 y4

【用法示例】:
    # 默认读取当前目录下 images/ 与 labels/，转换输出到 labels_obb/
    python msra2yolo_obb.py

    # 自定义路径与参数 (过滤困难样本, 自定义类别ID):
    python msra2yolo_obb.py --images path/to/images --labels path/to/labels --output path/to/labels_obb --cls-id 0 --filter-difficult
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import time
from pathlib import Path
from PIL import Image

# 支持的常见图像扩展名
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def clip(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """将数值限制在 [min_val, max_val] 区间内。"""
    return max(min_val, min(max_val, val))


def get_image_size(image_path: str | Path) -> tuple[int, int] | None:
    """
    高效获取图像宽高 (width, height)。
    仅读取图像文件头元数据，速度极快 (<1ms)，无需加载全图像素。
    """
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception:
        # 如 PIL 读取失败，尝试用 cv2 解码兜底
        try:
            import cv2
            import numpy as np
            raw_data = np.fromfile(str(image_path), dtype=np.uint8)
            img = cv2.imdecode(raw_data, cv2.IMREAD_UNCHANGED)
            if img is not None:
                h, w = img.shape[:2]
                return w, h
        except Exception:
            pass
    return None


def find_matched_image(label_stem: str, images_dir: str | Path) -> Path | None:
    """
    根据标签文件的主文件名匹配 images 目录下的图像文件 (不区分大小写)。
    """
    img_dir_path = Path(images_dir)
    # 直接探测常见后缀
    for ext in IMG_EXTENSIONS:
        candidate = img_dir_path / f"{label_stem}{ext}"
        if candidate.is_file():
            return candidate
        candidate_upper = img_dir_path / f"{label_stem}{ext.upper()}"
        if candidate_upper.is_file():
            return candidate_upper

    # 若未匹配到，则扫描 images 目录全名对比 (兼容特殊命名)
    label_stem_lower = label_stem.lower()
    for item in img_dir_path.iterdir():
        if item.is_file() and item.suffix.lower() in IMG_EXTENSIONS:
            if item.stem.lower() == label_stem_lower:
                return item
    return None


def rbox_to_polygon_corners(x: float, y: float, w: float, h: float, theta: float) -> list[tuple[float, float]]:
    """
    依据 MSRA-TD500 官方几何定义，将 (x, y, w, h, theta) 转为 4 个多边形顶点坐标 (顺时针/逆时针四边形)。
    - (x, y) 为未旋转时水平包围盒的左上角；
    - 中心点 (cx, cy) = (x + w/2, y + h/2)；
    - 绕中心点旋转 theta 弧度。
    """
    cx = x + w / 2.0
    cy = y + h / 2.0

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # 4 个角点相对中心 (cx, cy) 的偏移量
    # 顺序: 左上, 右上, 右下, 左下
    corner_offsets = [
        (-w / 2.0, -h / 2.0),
        ( w / 2.0, -h / 2.0),
        ( w / 2.0,  h / 2.0),
        (-w / 2.0,  h / 2.0),
    ]

    corners = []
    for dx, dy in corner_offsets:
        # 旋转变换公式:
        # px = cx + dx * cos(theta) - dy * sin(theta)
        # py = cy + dx * sin(theta) + dy * cos(theta)
        px = cx + dx * cos_t - dy * sin_t
        py = cy + dx * sin_t + dy * cos_t
        corners.append((px, py))

    return corners


def convert_msra_to_yolo_obb(
    labels_dir: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    cls_id: int = 0,
    filter_difficult: bool = True,
    precision: int = 6,
) -> None:
    """
    将 MSRA-TD500 标签文件夹批量转换为 YOLO26 OBB 格式。
    """
    labels_path = Path(labels_dir)
    images_path = Path(images_dir)
    output_path = Path(output_dir)

    if not labels_path.exists():
        raise FileNotFoundError(f"标签目录不存在: {labels_path.resolve()}")
    if not images_path.exists():
        raise FileNotFoundError(f"图像目录不存在: {images_path.resolve()}")

    output_path.mkdir(parents=True, exist_ok=True)

    # 收集所有的 .gt 和 .txt 标签文件
    label_files = sorted(
        [p for p in labels_path.iterdir() if p.is_file() and p.suffix.lower() in (".gt", ".txt")]
    )

    if not label_files:
        print(f"[警告] 在 {labels_path.resolve()} 中未检索到 .gt 或 .txt 标签文件！")
        return

    print("=" * 70)
    print("开始转换 MSRA-TD500 标注 -> YOLO26 OBB 格式:")
    print(f"  * 标签源目录: {labels_path.resolve()} (共检索到 {len(label_files)} 个标注文件)")
    print(f"  * 图像源目录: {images_path.resolve()}")
    print(f"  * 输出目标目录: {output_path.resolve()}")
    print(f"  * 目标类别 ID: {cls_id}")
    print(f"  * 过滤 difficult 样本 (difficult_flag==1): {'是' if filter_difficult else '否 (全部保留)'}")
    print(f"  * 归一化坐标小数精度: {precision} 位")
    print("=" * 70)

    start_time = time.time()
    num_converted_files = 0
    num_total_boxes = 0
    num_difficult_skipped = 0
    num_missing_images = 0

    fmt = f"{{:.{precision}f}}"

    for lbl_file in label_files:
        stem = lbl_file.stem
        # 联动对应的图像文件
        img_file = find_matched_image(stem, images_path)
        if img_file is None:
            print(f"[提示] 未找到标签 {lbl_file.name} 对应的原图，跳过。")
            num_missing_images += 1
            continue

        # 获取图像宽高用于坐标归一化
        img_size = get_image_size(img_file)
        if img_size is None or img_size[0] <= 0 or img_size[1] <= 0:
            print(f"[警告] 无法解析图像尺寸: {img_file.name}，跳过该文件。")
            continue

        img_w, img_h = img_size
        out_lines: list[str] = []

        # 读取并解析标注文件内容
        try:
            with open(lbl_file, "r", encoding="utf-8-sig", errors="ignore") as f:
                raw_lines = f.readlines()
        except Exception as e:
            print(f"[错误] 读取文件 {lbl_file.name} 失败: {e}")
            continue

        for line in raw_lines:
            line_str = line.strip()
            if not line_str:
                continue

            parts = line_str.split()
            # MSRA-TD500 格式: index difficulty_flag x y w h theta
            if len(parts) < 7:
                continue

            try:
                # 解析字段
                _, diff_flag, x_val, y_val, w_val, h_val, theta_val = parts[:7]
                diff = int(diff_flag)
                x = float(x_val)
                y = float(y_val)
                w = float(w_val)
                h = float(h_val)
                theta = float(theta_val)
            except ValueError:
                continue

            # 判断是否过滤困难样本 (如模糊、严重遮挡)
            if filter_difficult and diff == 1:
                num_difficult_skipped += 1
                continue

            # 将 (x, y, w, h, theta) 转为 4 个顶点坐标
            corners = rbox_to_polygon_corners(x, y, w, h, theta)

            # 将 4 个顶点的像素坐标归一化到 [0, 1]
            norm_coords: list[float] = []
            for px, py in corners:
                nx = clip(px / img_w, 0.0, 1.0)
                ny = clip(py / img_h, 0.0, 1.0)
                norm_coords.extend([nx, ny])

            # 组装为 YOLO OBB 官方标准格式: class_id x1 y1 x2 y2 x3 y3 x4 y4
            coord_str = " ".join(fmt.format(c) for c in norm_coords)
            out_lines.append(f"{cls_id} {coord_str}")
            num_total_boxes += 1

        # 写入转换后的 YOLO 标准 txt 标签 (以原图同名命名)
        out_file = output_path / f"{stem}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            for l in out_lines:
                f.write(l + "\n")

        num_converted_files += 1

    elapsed = time.time() - start_time
    print("-" * 70)
    print("转换完毕！统计结果:")
    print(f"  * 成功生成标签文件数: {num_converted_files}")
    print(f"  * 转换有效目标框总数: {num_total_boxes}")
    if filter_difficult:
        print(f"  * 过滤 difficult 框数: {num_difficult_skipped}")
    if num_missing_images > 0:
        print(f"  * 未找到匹配图像的文件数: {num_missing_images}")
    print(f"  * 总耗时: {elapsed:.2f} 秒")
    print(f"  * 保存位置: {output_path.resolve()}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="MSRA-TD500 标注转 YOLO26 OBB 格式工具 (自动联动图像尺寸进行归一化)"
    )
    parser.add_argument(
        "--labels", default="labels", type=str, help="MSRA-TD500 标签文件夹路径 (存放 .gt 或 .txt)"
    )
    parser.add_argument(
        "--images", default="images", type=str, help="原图文件夹路径 (存放 .JPG / .png 等)"
    )
    parser.add_argument(
        "--output", default="labels_obb", type=str, help="输出 YOLO 标签文件夹路径 (默认: labels_obb)"
    )
    parser.add_argument(
        "--cls-id", default=0, type=int, help="YOLO 文本类别 ID (默认: 0)"
    )
    parser.add_argument(
        "--filter-difficult",
        action="store_true",
        default=True,
        help="是否过滤 difficulty_flag 为 1 的难例样本 (默认开启)",
    )
    parser.add_argument(
        "--keep-difficult",
        action="store_false",
        dest="filter_difficult",
        help="保留所有 difficulty 样本，不作过滤",
    )
    parser.add_argument(
        "--precision", default=6, type=int, help="坐标归一化保留的小数位数 (默认: 6)"
    )

    args = parser.parse_args()

    convert_msra_to_yolo_obb(
        labels_dir=args.labels,
        images_dir=args.images,
        output_dir=args.output,
        cls_id=args.cls_id,
        filter_difficult=args.filter_difficult,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()
