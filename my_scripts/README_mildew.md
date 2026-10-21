# 白粉病二分类与阈值实验：服务器运行

服务器压缩包 `mildew_resnet18_server.zip` 包含 Ultralytics 8.4.174 源码、实验脚本、小麦四分类检测器，以及已经训练好的 ResNet18 白粉病二分类器。无需重新训练。原始四分类检测数据集需单独上传；从 Git 仓库获取代码时，还需自行复制下列权重。

| 文件                                    | 内容                                                              |
| --------------------------------------- | ----------------------------------------------------------------- |
| `weights/wheat_detector_best.pt`        | 本地 `v8n_four_classes_200epochs-4/weights/best.pt`               |
| `weights/mildew_resnet18_best.pt`       | `wheat_mildew/runs/20261021_084829_579500/best.pt`，最佳 epoch 21 |
| `weights/resnet18_training_config.json` | 分类训练参数，记录来源；其中 Windows 路径不参与服务器推理         |
| `weights/resnet18_test_metrics.json`    | 原分类测试集指标，不等于级联检测结果                              |
| `data.example.yaml`                     | 服务器检测数据配置模板                                            |

## 1. 环境与数据

解压后进入根目录，建议 Python 3.11，使用匹配的 PyTorch 2.x / torchvision 和服务器 CUDA 环境。不要复制本地 Windows conda 环境。

```bash
unzip mildew_resnet18_server.zip
cd mildew_resnet18_server
python -m pip install -e .
python -c 'import ultralytics, torch, torchvision; print(ultralytics.__version__, torch.__version__, torchvision.__version__, torch.cuda.is_available())'
```

上传原来的 `dataset_four_classes`，保留 `images/{train,val,test}` 和 `labels/{train,val,test}`。将 `data.example.yaml` 复制为 `data.server.yaml`，修改其中 `path` 为服务器上的数据目录绝对路径；类别顺序必须保持 BrownRust、Healthy、Loose-Smut、Mildew。

## 2. 直接运行四组阈值与对照

从项目根目录运行：

```bash
python my_scripts/mildew_refinement.py evaluate \
  --model weights/wheat_detector_best.pt \
  --classifier weights/mildew_resnet18_best.pt --classifier-type resnet18 \
  --data data.server.yaml --device 0 \
  --thresholds 0.15 0.20 0.30 0.35 --cls-thresholds 0.50 \
  --split val --output outputs/resnet18_val
```

`--output` 必须是尚不存在的目录；重跑时更换名称。无 GPU 可使用 `--device cpu`，显存不足可增加 `--cls-batch 8`。

本轮共九个工作点：四个白粉病检测阈值各运行有/无分类过滤的对照，再增加一个原始白粉病阈值 0.25、无二分类器的基线。分类器只审核预测类别为 Mildew 的框：白粉病分数达到 0.50 则保留，否则删除。其他三类不进入分类器，检测阈值固定 0.25；脚本逐图检查这些类别的输出在各组完全一致。

检测阈值与二分类阈值是两个参数。分数不相乘，不改原框坐标和类别。检测固定输入 640、类别独立 NMS、NMS IoU=0.70、max_det=300、方形预处理。每张图检测一次、候选框批量分类一次，各工作点共享结果；若输出提示 max_det 截断，应检查截断对对照的影响。

ResNet18 对框四周各扩展宽高的 10%，取完整区域，转 RGB，使用与其训练一致的 `ImageOps.pad(224×224)`、双线性缩放、RGB 填充色 `(124,116,104)` 和 ImageNet 标准化。读取 softmax 中 `powdery_mildew` 的分数，不使用 top1 置信度。推理不下载预训练权重。

## 3. 查看结果

- `outputs/resnet18_val/summary.csv`：每组四类的 TP、FP、FN、Precision、Recall、F1、FP/图和候选框数/图。按类别 Mildew 筛选即可查看九组。
- `predictions.jsonl`：逐图 GT、原检测框、检测分数和二分类白粉病分数。
- `config.json`：实际参数、版本和两个权重的 SHA256。
- `timing.json`：共享推理耗时与截断图片数，含首次冷启动，不能当各工作点的部署 FPS。

匹配复用 Ultralytics 类别一致、IoU 优先的一对一匹配，匹配 IoU=0.50。此处报告固定阈值指标，不直接与训练截图里的 mAP 或默认混淆矩阵比较。可独立核对无分类器的 0.25 基线：

```bash
python my_scripts/mildew_refinement.py evaluate \
  --model weights/wheat_detector_best.pt --data data.server.yaml --device 0 \
  --thresholds 0.25 --split val --output outputs/direct_025_val
```

在 val 上选择阈值，冻结后再运行 test。例如选定检测阈值 0.20、二分类阈值 0.50 后，将第 2 步改为 `--thresholds 0.20 --cls-thresholds 0.50 --split test --output outputs/resnet18_test`，同时得到该阈值无分类器对照和 0.25 基线。不要根据 test 反复调参。

现有 ResNet18 在 WFD 整图上训练，应用到检测裁剪框存在分布差异；原分类测试集 Precision=0.5833、Recall=0.6364、F1=0.6087。过滤是否有效，应由本轮整图检测 val 结果判断，同时看减少的误报和损失的真阳性。仅降低 FP 不足以证明方案更好。

## 4. 可选：沿用候选审核及 YOLO 二分类流程

现有 `prepare` 和 `build` 命令仍可使用，本次服务器运行无需执行这两步。

```bash
python my_scripts/mildew_refinement.py prepare \
  --model weights/wheat_detector_best.pt --data data.server.yaml --device 0 \
  --output outputs/mildew_candidates

python my_scripts/mildew_refinement.py build \
  --review outputs/mildew_candidates/review.csv --output data/mildew_binary
```

prepare 仅使用检测 train，以 0.10 导出白粉病候选，扩边 10% 后补灰色正方形。查看 crops，在 CSV 的 `reviewed_label` 填 `mildew`、`non_mildew` 或 `skip`；空白与 skip 不参与构建，建议标签不会自动用于训练。无匹配标注的框不一定是真背景，仍需看图确认。

默认按源图片分组，去除常见 `.rf.` 后缀，再以 seed=42 的稳定哈希约 80:20 划分分类 train/val。若同植株、序列或增强版本的命名无法自动合并，prepare 增加 `--groups groups.csv`，提供覆盖检测 train 全部文件名的 `image,group` CSV。build 检查分组不跨分类 split，且每侧正负类至少两个审核样本；这只是可运行下限。

由此训练的 YOLO 分类权重仍可用 `--classifier-type yolo` 接入，要求两类名恰为 `mildew` 和 `non_mildew`。此路径保留原有正方形裁剪和 YOLO 分类预处理，与当前 ResNet18 权重格式不同。
