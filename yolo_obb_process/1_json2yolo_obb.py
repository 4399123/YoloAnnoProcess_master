"""
JSON (X-AnyLabeling / LabelMe 风格) -> Ultralytics YOLO OBB 标签转换工具

输出格式 (Ultralytics OBB 官方规范, 每个目标一行):
    class_index x1 y1 x2 y2 x3 y3 x4 y4
所有坐标均按图像宽高归一化到 [0, 1]。

输出目录结构:
    <out_dir>/images/xxx.png   原图
    <out_dir>/labels/xxx.txt   标签

类别过滤 (参考 2_voc_label.py 对 Other 类的处理):
    - 不在 CLASSES 列表中的类别直接跳过;
    - "Other" 类即使在列表中也会被过滤掉, 不写入标签。
    - 建议把 "Other" 放在列表最后, 这样其余类别的索引是连续的 0..N-1。

用法示例:
    python json2yolo_obb.py
    python json2yolo_obb.py --input o_imgs --output . --classes BS Other
    python json2yolo_obb.py --skip-difficult --move
"""

import argparse
import json
import os
import shutil

# 默认需要转换的类别列表 (可通过 --classes 覆盖)
# CLASSES = ["car", "person", "Other"]
# CLASSES = ["BS", "Other"]
CLASSES = ["wenzi", "Other"]

FILTER_CLASS = "Other"  # 需要过滤的类别名
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def clip01(v):
    return min(max(v, 0.0), 1.0)


def shape_to_4points(shape):
    """把一个标注 shape 转为 4 个角点 [(x, y), ...], 无法转换时返回 None。"""
    pts = shape.get("points") or []
    stype = shape.get("shape_type", "polygon")

    if stype in ("rotation", "polygon") and len(pts) == 4:
        return [(float(p[0]), float(p[1])) for p in pts]

    if stype == "rectangle":
        if len(pts) == 4:
            return [(float(p[0]), float(p[1])) for p in pts]
        if len(pts) == 2:
            (xa, ya), (xb, yb) = pts
            x1, x2 = sorted((float(xa), float(xb)))
            y1, y2 = sorted((float(ya), float(yb)))
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    return None


def find_image(json_path, data, input_dir):
    """根据 imagePath 或同名文件查找原图路径。"""
    image_path = data.get("imagePath")
    if image_path:
        cand = os.path.join(input_dir, os.path.basename(image_path))
        if os.path.isfile(cand):
            return cand
    stem = os.path.splitext(json_path)[0]
    for ext in IMG_EXTS:
        for e in (ext, ext.upper()):
            if os.path.isfile(stem + e):
                return stem + e
    return None


def get_image_size(data, img_path):
    w, h = data.get("imageWidth"), data.get("imageHeight")
    if w and h:
        return int(w), int(h)
    # JSON 中缺少尺寸信息时再读取图片 (需要 Pillow)
    from PIL import Image
    with Image.open(img_path) as im:
        return im.size


def convert_one(json_path, input_dir, images_dir, labels_dir, classes,
                filter_id, skip_difficult, move, stats):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_path = find_image(json_path, data, input_dir)
    if img_path is None:
        print(f"[警告] 找不到对应图片, 跳过: {json_path}")
        stats["missing_img"] += 1
        return

    w, h = get_image_size(data, img_path)
    lines = []
    for shape in data.get("shapes", []):
        cls = shape.get("label")
        # 不在类别列表中的类别不参与转换
        if cls not in classes:
            stats["skipped_cls"] += 1
            continue
        cls_id = classes.index(cls)
        # 过滤 Other 类
        if cls_id == filter_id:
            stats["skipped_cls"] += 1
            continue
        if skip_difficult and shape.get("difficult"):
            stats["skipped_difficult"] += 1
            continue

        pts = shape_to_4points(shape)
        if pts is None:
            print(f"[警告] 不支持的标注 (shape_type={shape.get('shape_type')}, "
                  f"点数={len(shape.get('points') or [])}), 跳过: {json_path}")
            stats["skipped_shape"] += 1
            continue

        coords = []
        for x, y in pts:
            coords.append(f"{clip01(x / w):.6f}")
            coords.append(f"{clip01(y / h):.6f}")
        lines.append(f"{cls_id} " + " ".join(coords))

    stem = os.path.splitext(os.path.basename(img_path))[0]
    # 没有有效目标时也生成空 txt, 作为背景图参与训练
    with open(os.path.join(labels_dir, stem + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    dst_img = os.path.join(images_dir, os.path.basename(img_path))
    if os.path.abspath(img_path) != os.path.abspath(dst_img):
        (shutil.move if move else shutil.copy2)(img_path, dst_img)

    stats["files"] += 1
    stats["objects"] += len(lines)


def main():
    parser = argparse.ArgumentParser(description="JSON 旋转框标注 -> Ultralytics YOLO OBB txt")
    parser.add_argument("--input", default="o_imgs", help="输入文件夹 (JSON + 原图)")
    parser.add_argument("--output", default=".", help="输出根目录, 将生成 images/ 和 labels/")
    parser.add_argument("--classes", nargs="+", default=CLASSES,
                        help="需要转换的类别列表 (顺序即 class_index), 'Other' 会被过滤")
    parser.add_argument("--skip-difficult", action="store_true", help="跳过 difficult=true 的目标")
    parser.add_argument("--move", action="store_true", help="移动原图而不是复制")
    args = parser.parse_args()

    classes = list(args.classes)
    filter_id = classes.index(FILTER_CLASS) if FILTER_CLASS in classes else -1

    images_dir = os.path.join(args.output, "images")
    labels_dir = os.path.join(args.output, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    stats = dict(files=0, objects=0, skipped_cls=0, skipped_difficult=0,
                 skipped_shape=0, missing_img=0)

    json_files = sorted(f for f in os.listdir(args.input) if f.lower().endswith(".json"))
    for name in json_files:
        convert_one(os.path.join(args.input, name), args.input, images_dir, labels_dir,
                    classes, filter_id, args.skip_difficult, args.move, stats)

    names = [c for i, c in enumerate(classes) if i != filter_id]
    print(f"类别映射: {{{', '.join(f'{classes.index(c)}: {c}' for c in names)}}}")
    print(f"完成: {stats['files']} 张图片, {stats['objects']} 个目标; "
          f"过滤类别 {stats['skipped_cls']}, difficult {stats['skipped_difficult']}, "
          f"不支持形状 {stats['skipped_shape']}, 缺图 {stats['missing_img']}")
    print(f"输出: {os.path.abspath(images_dir)} | {os.path.abspath(labels_dir)}")


if __name__ == "__main__":
    main()
