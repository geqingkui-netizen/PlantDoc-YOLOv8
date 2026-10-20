# 白粉病二分类与阈值实验：服务器运行

服务器打包版包含同版本 Ultralytics 源码、实验脚本、当前小麦检测器和 ImageNet 分类预训练权重，不包含原始数据集，也不包含已经训练好的白粉病二分类器。`weights/wheat_detector_best.pt` 来自本地 `runs/wheat/v8n_four_classes_200epochs-4/weights/best.pt`；`weights/yolov8n-cls.pt` 是通用分类预训练权重，需要用自己的正负样本训练。若从 Git 仓库获取脚本而非服务器压缩包，请自行复制这两个权重到上述路径。

## 1. 环境与数据

解压后进入项目根目录，使用已有 CUDA/PyTorch 环境。建议 Python 3.11；PyTorch 与 torchvision 必须匹配服务器 CUDA 驱动。不要直接复制本地 Windows 的 conda 环境。

```bash
cd mildew_server
python -m pip install -e .
python -c 'import ultralytics, torch; print(ultralytics.__version__, torch.__version__, torch.cuda.is_available())'
python my_scripts/mildew_refinement.py --help
```

把原来的四分类数据集复制到服务器，保持 images/labels 中的 train、val、test 划分，并修改 data.yaml 的 path 为服务器绝对路径，例如：

```yaml
path: /data/dataset_four_classes
train: images/train
val: images/val
test: images/test
nc: 4
names:
  0: BrownRust
  1: Healthy
  2: Loose-Smut
  3: Mildew
```

以下命令从项目根目录运行。设置你的数据配置路径：

```bash
DATA=/data/dataset_four_classes/data.yaml
MODEL=weights/wheat_detector_best.pt
```

每个 --output 必须是尚不存在的新目录，避免覆盖已审核标签或实验结果。重跑请更换目录名。初次建议先跑第 2 步，确认服务器环境和数据可用。

## 2. 立即可跑：四个检测阈值的基线

```bash
python my_scripts/mildew_refinement.py evaluate \
  --model "$MODEL" --data "$DATA" --device 0 \
  --split val --output outputs/baseline_val
```

不需要二分类器即可运行。本步比较白粉病阈值 0.25、0.20、0.15、0.10；其他类别阈值固定 0.25。结果在 `outputs/baseline_val/summary.csv`，每个工作点分别列出四类的 TP、FP、FN、Precision、Recall、F1 和 FP/图。

输入固定 640，NMS IoU=0.70，max_det=300，类别独立 NMS。这里使用统一的方形 predict 预处理，与原训练日志的验证管线不完全相同；所有对照均由本脚本重新计算，不能直接拿截图里的日志数值当固定阈值基线。

## 3. 采集白粉病候选并审核标签

```bash
python my_scripts/mildew_refinement.py prepare \
  --model "$MODEL" --data "$DATA" --device 0 \
  --output outputs/mildew_candidates
```

本步只读取检测 train，检测阈值 0.10，只裁剪预测为 Mildew 的框。框四周各扩展宽高的 10%，补灰边为正方形，保存完整裁剪供训练和推理使用。

输出：

- `outputs/mildew_candidates/crops/`：需要查看的裁剪 PNG。
- `outputs/mildew_candidates/review.csv`：候选清单，含原图、裁剪路径、分组、分类 train/val、检测分数、与白粉病标注的最大 IoU、建议标签、人工审核标签。
- `outputs/mildew_candidates/config.json`：参数、框架版本和检测权重 SHA256。

打开 review.csv，查看对应 crops，在 **reviewed_label** 列填写以下之一，然后保存为 UTF-8 CSV：

| 值         | 含义                                 |
| ---------- | ------------------------------------ |
| mildew     | 裁剪中确实可辨识白粉病               |
| non_mildew | 白粉病误报：健康组织、其他疾病或背景 |
| skip       | 不确定、混合症状或暂不用于训练       |

空白表示尚未审核，构建分类数据时会排除。**建议标签不会自动进入训练。** 无匹配框、重复框或定位偏差框仍可能包含真实白粉病，不能直接把全部 FP 当负样本。人工判断应以实际保存裁剪为准。应覆盖低置信度真阳性和不同置信度段的误报，不要只审核最容易的样本。

默认按源图片分组，去掉常见 `.rf.` 后缀来合并同源 Roboflow 版本，再以 seed=42 的稳定哈希约 80:20 划分分类 train/val。默认不能识别全部增强命名或植株身份。如果存在同植株/同序列/增强图片，请提供完整分组 CSV：

```csv
image,group
Mildew_1.jpg,plant_001
Mildew_1_aug.jpg,plant_001
Healthy_2.jpg,plant_002
```

prepare 命令增加 `--groups /path/to/groups.csv`；CSV 必须覆盖检测 train 的全部文件名。分组应在审核前确定。所有二分类样本均来自检测 train；检测 val/test 不用于分类训练。检测器见过这些训练图，因此内部分类验证结果可能乐观，最终以整图 val/test 为准。

## 4. 构建审核后的分类数据

```bash
python my_scripts/mildew_refinement.py build \
  --review outputs/mildew_candidates/review.csv \
  --output data/mildew_binary
```

输出 `data/mildew_binary/{train,val}/{mildew,non_mildew}/`。脚本检查源分组不跨分类 train/val，两侧各有至少两个已审核正、负样本；这个下限只是防止无法训练，不代表充分的样本量。尽量审核有代表性的足量样本，避免大量近重复裁剪占据训练集。

## 5. 训练一个二分类器

```bash
yolo classify train \
  model=weights/yolov8n-cls.pt data=data/mildew_binary \
  imgsz=224 epochs=100 patience=20 batch=32 \
  device=0 workers=4 seed=42 deterministic=True \
  scale=0.0 auto_augment=None erasing=0.0 \
  fliplr=0.5 flipud=0.0 hsv_h=0.0 hsv_s=0.1 hsv_v=0.1 \
  project="$PWD/runs/mildew_binary" name=v8n_seed42
```

如果服务器已有预训练权重，model 改为该文件路径。显存不足可降低分类 batch。project 使用绝对路径，避免训练结果被写入全局 runs_dir 的其他子目录。关闭随机裁剪面积变化、随机擦除和自动增强，保持补成正方形的候选区域完整。一个二分类器用于所有阈值组，不为每个阈值重新训练。

训练结束后记住日志实际输出目录；下面假定是 `runs/mildew_binary/v8n_seed42/weights/best.pt`，如自动递增了目录名，应使用实际路径。

## 6. 八组完整验证对照

```bash
CLS=runs/mildew_binary/v8n_seed42/weights/best.pt
python my_scripts/mildew_refinement.py evaluate \
  --model "$MODEL" --data "$DATA" --classifier "$CLS" --device 0 \
  --split val --output outputs/cascade_val
```

默认四个白粉病检测阈值，每个阈值分别测试有/无二分类器，共八组。分类阈值固定 0.50，只比较分类器 **mildew 类分数**，不拿 top1conf 替代。分类器只保留或删除白粉病框；其他三类不进入分类器，固定 0.25，脚本检查各组其他三类输出一致。

每图检测一次，白粉病候选分批分类一次，所有工作点共用结果。首轮保留检测原分数，不相乘、不改类别、不改原框坐标。输出：

- `summary.csv`：固定工作阈值下四类的指标；匹配直接复用 Ultralytics 类别一致、IoU 优先的一对一匹配，匹配 IoU=0.50。
- `predictions.jsonl`：原图路径、GT、检测框、检测分数和二分类白粉病分数，可追溯每个预测。
- `config.json`：运行参数、版本、检测和分类权重 SHA256。
- `timing.json`：本轮共享推理实际耗时与 max_det 截断图片数。含首次冷启动，不是每个工作点的部署延迟；不能据此直接宣称某组 FPS。

为独立核对 0.25 推理得到的基线，还可以单跑：

```bash
python my_scripts/mildew_refinement.py evaluate \
  --model "$MODEL" --data "$DATA" --device 0 --thresholds 0.25 \
  --split val --output outputs/direct_025_val
```

与八组中的无二分类器 0.25 组对照。若不一致，先检查 NMS、max_det、环境与数据变化，不把差异归因为分类器。

## 7. 在 val 选阈值，冻结后再测 test

第一轮先看 tau=0.50 的八组，建议预先固定选择规则：白粉病 Recall 相比原始 0.25 基线下降不超过 1 个百分点时，优先选择 FP 最少者；并列时选 F1 更高者。1 个百分点是实验容忍值，不是已得出的结论。

如果需要进一步选分类阈值，只在 val 上运行：

```bash
python my_scripts/mildew_refinement.py evaluate \
  --model "$MODEL" --data "$DATA" --classifier "$CLS" --device 0 \
  --cls-thresholds 0.30 0.40 0.50 0.60 0.70 0.80 \
  --split val --output outputs/cascade_val_tau_grid
```

得到 4 个无分类对照加 24 个级联工作点。不要根据 test 选择阈值。假定仅举例说明，val 最终选出检测 0.15、分类 0.40，则：

```bash
python my_scripts/mildew_refinement.py evaluate \
  --model "$MODEL" --data "$DATA" --device 0 --thresholds 0.25 \
  --split test --output outputs/baseline_test

python my_scripts/mildew_refinement.py evaluate \
  --model "$MODEL" --data "$DATA" --classifier "$CLS" --device 0 \
  --thresholds 0.15 --cls-thresholds 0.40 \
  --split test --output outputs/selected_test
```

第二条同时输出同检测阈值 0.15 的无分类对照和级联结果。上面的 0.15/0.40 是命令示例，不是已选定的最优值。最后确认其他类别保持一致，白粉病召回约束满足；若未满足，如实报告。

本脚本主要报告固定工作点 P/R/F1/FP/FN，不把截断候选后的 AP 与原训练 mAP 混作同口径比较，也不会自动运行测试集。完整部署延迟需在冻结配置下另做含裁剪与两个模型的热身、同步计时。
