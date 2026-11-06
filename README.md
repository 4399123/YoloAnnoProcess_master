# YOLO 数据集处理工具

本仓库用于把标注工具导出的标注文件，转换成 YOLO 训练所需的数据集。内容分为两部分：

- `yolo_hbb_process/`：常规水平框目标检测（HBB，Horizontal Bounding Box），标注为 VOC 格式 xml，输出 YOLO 格式的 txt 标签和 train/val 数据集。
- `yolo_obb_process/`：旋转框目标检测（OBB，Oriented Bounding Box），标注为 LabelMe / X-AnyLabeling 风格的 json，输出 Ultralytics YOLO OBB 格式的标签和数据集。

所有脚本都使用相对路径，需要在对应的数据目录下运行。

## 环境依赖

```
python >= 3.7
opencv-python
numpy
tqdm
imutils          # 仅 HBB 流程使用
Pillow           # 仅 OBB 流程中 json 缺少图像尺寸时使用
```

## 一、常规 YOLO 目标检测（HBB）

输入为 VOC 格式的 xml 标注（`Annotations/`）和同名的 png 原图。脚本按以下顺序执行，建议在 `yolo_hbb_process/` 目录下运行。

| 脚本 | 作用 |
| --- | --- |
| `0_copy_imgs_out.py` | 从 `o_imgs/` 中把与 json 同名的 png 图片复制到 `images/` |
| `1_split_train_val.py` | 读取 `Annotations/` 下的 xml，按 9:1 随机划分，生成 `ImageSets/train.txt` 和 `ImageSets/val.txt` |
| `2_voc_label.py` | 读取 xml 和 `ImageSets/` 中的划分列表，转换为 YOLO 格式写入 `labels/`，并生成根目录下的 `train.txt`、`val.txt` |
| `3_验证标注准确性.py` | 把 `labels/` 中的框画回 `images/` 的原图，输出到 `suoying_output/`，用于核对标注。框上标注类别 id |
| `3_验证标注准确性2.py` | 同上，但框上标注的是该图中第几个目标的序号，便于逐个检查 |
| `4_get_yolov6_formatdataset.py` | 按 `train.txt`、`val.txt` 把图片和标签整理到 `out/images/{train,val}` 与 `out/labels/{train,val}` |
| `扣取目标.py` | 根据 `out/` 中的标签，把每个目标框从原图裁剪出来，保存到 `cut/` |

处理流程要点：

1. 起点是已经生成好的 VOC xml 标注，本仓库不包含由 json 转 xml 的步骤。
2. `2_voc_label.py` 中的类别列表是写死在代码里的（`classes`），切换数据集时需要修改。`Other` 类和 `difficult=1` 的目标会被过滤掉。
3. 归一化格式为 `class x_center y_center width height`，坐标均除以图像宽高。
4. 第 4 步依赖第 2 步生成的根目录 `train.txt` 和 `val.txt`，两者中的文件名都是 png 图片名。

## 二、旋转框目标检测（OBB）

输入为 LabelMe / X-AnyLabeling 风格的 json（`o_imgs/` 中与图片同名的 json），标注形状为 4 点的 polygon、rotation 或 rectangle。

| 脚本 | 作用 |
| --- | --- |
| `1_json2yolo_obb.py` | 解析 json，转换为 YOLO OBB 标签，输出到 `images/` 和 `labels/` |
| `2_resize_dataset.py` | 把图像直接拉伸到固定分辨率（默认 800x800，不加黑边），同时换算标签坐标，输出到 `imagesX/` 和 `labelsX/` |
| `3_visualize_obb.py` | 把 `imagesX/` 和 `labelsX/` 中的旋转框画到图上，输出到 `suoying_output/`，用于核对 |
| `4_split_dataset.py` | 按比例划分训练集和验证集，输出到 `dataset/images/{train,val}` 与 `dataset/labels/{train,val}` |

OBB 标签格式（Ultralytics 官方规范，每个目标一行）：

```
class_index x1 y1 x2 y2 x3 y3 x4 y4
```

所有坐标都按图像宽高归一化到 [0, 1]。

推荐执行顺序：

```
python 1_json2yolo_obb.py --input o_imgs --output . --classes <类别...> Other
python 2_resize_dataset.py --W 800 --H 800 --images images --labels labels --out_images imagesX --out_labels labelsX
python 3_visualize_obb.py --images imagesX --labels labelsX --output suoying_output --names <类别...>
python 4_split_dataset.py --images imagesX --labels labelsX --output dataset --train_percent 0.9
```

注意事项：

- `--classes` 的顺序就是 `class_index` 的顺序，`3_visualize_obb.py` 的 `--names` 必须与之保持一致。`Other` 类会被过滤，建议放在列表最后。
- 没有目标的图片也会输出空的 txt 文件，作为背景图参与训练。
- `2_resize_dataset.py` 对空标签的图片只做图像缩放，并输出空 txt，保证图片和标签一一对应。
- `1_json2yolo_obb.py` 支持 `--skip-difficult` 跳过困难样本，`--move` 改为移动原图而不是复制。
- 不支持的标注形状会被打印警告并跳过。

## 目录结构示例

```
yolo_hbb_process/
├── o_imgs/            # 原始图片和 json
├── Annotations/       # VOC xml 标注
├── ImageSets/         # 划分结果 train.txt / val.txt
├── images/            # 整理后的图片
├── labels/            # YOLO 格式标签
├── out/               # 4_get_yolov6_formatdataset.py 的输出
└── cut/               # 扣取目标的输出

yolo_obb_process/
├── o_imgs/            # 原始图片和 json
├── images/ labels/    # 1_json2yolo_obb.py 的输出
├── imagesX/ labelsX/  # 2_resize_dataset.py 的输出
├── suoying_output/    # 可视化结果
└── dataset/           # 4_split_dataset.py 的输出
```

## 注意

- 脚本中包含的类别列表、路径和划分比例都是写死的默认值或命令行参数，使用前请先检查。
- 处理前建议先用可视化脚本抽查标注，确认无误后再进入训练。
