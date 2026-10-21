# 三层区域审核分支：五类训练、四类检测输出

在 PyCharm 中打开本文件所在的 `wheat_fourclass_joint` 文件夹。Python 解释器沿用：

```text
E:\Programes\ANACONDA\envs\pytorch311\python.exe
```

在项目根目录的 Terminal 中运行：

```powershell
.\start_local_training.bat
```

默认使用本机 `E:/Dataset/dataset_four_classes/data.yaml`，YOLOv8n 预训练权重、100 轮、640 输入、batch=2、workers=0、seed=42、区域损失权重 0.5。启动前请等其他训练结束，避免占满 RTX 3050 的 4 GB 显存。可覆盖参数，例如：

```powershell
.\start_local_training.bat --epochs 200 --batch 1 --name fourclass_200_seed42
```

新分类分支随机初始化，与 YOLO 一起学习。包内 `weights/yolov8n.pt` 是预训练权重，不是已经训练好的小麦审核模型。此目录独立于原来的一层、三层二分类项目。

## 结构与输出规则

```text
图像 → YOLOv8 共享特征 ──→ 原四类检测头 → 类别独立 NMS → 保留或拒绝
                   └──→ P3 区域 ROIAlign（7×7）
                         → 3×3 卷积＋ReLU，64 通道
                         → 3×3 卷积＋ReLU，64 通道
                         → 3×3 卷积＋ReLU，64 通道
                         → 全局平均池化 → 全连接 → 五类 Softmax
```

| ID  | 检测与区域监督类别   | 是否输出检测框     |
| --- | -------------------- | ------------------ |
| 0   | BrownRust／褐锈病    | 是                 |
| 1   | Healthy／健康        | 是                 |
| 2   | Loose-Smut／散黑穗病 | 是                 |
| 3   | Mildew／白粉病       | 是                 |
| 4   | Background／背景     | 否，仅训练辅助分支 |

原 `data.yaml` 和检测标签保持四类，无需添加背景标签。健康目标和背景是不同的监督类别。

审核时，只有辅助分支的最高概率类别与 YOLO 原类别一致，且该类别概率达到审核阈值，才保留框。判为背景、判为其他类、概率不足，均删除该框。保留框的类别、坐标、检测置信度不改动；不会补框，也不会把白粉病框改成其他病害。审核概率与 YOLO 检测置信度是两个不同的值。

`--review-threshold 0` 完全关闭审核，用于同一份联合权重的有／无审核对照。默认训练审核阈值为 0.50，并随模型保存；独立验证、预测、断点续训沿用保存的设置，验证和预测可显式覆盖。阈值 0.50 是初始工作点，不代表已调优。

## 无需新增人工标注

每次直接从增强后的训练图片、已有 YOLO 检测标注生成区域监督：

- 已有 GT 框按真实类别监督，区域每侧扩展 10%，并裁剪到网络输入边界。
- 从全部四类的检测分数中取最高的至多 64 个候选，同样扩边后匹配 GT。
- 候选与某 GT 的 IoU≥0.50，且与其他类别没有 IoU≥0.20 的冲突匹配时，采用对应 GT 的类别。
- 候选区域与所有 GT 均没有交集时，监督为背景。
- 其余模糊候选跳过。YOLO 自己预测的类别不作为训练答案。

每张图每类最多选 32 个区域，先计算每个已出现类别的平均交叉熵，再对这些类别平均，避免背景数量压倒目标类。总损失为：

```text
原 YOLO 检测损失 + region_weight × 五类区域分类损失
```

区域损失同时更新三层分类器和共享特征；候选坐标及自动标签不求梯度。训练循环、检测匹配、优化器、EMA、保存和断点续训均复用原框架。

自动背景标签依赖原检测标注的完整性，漏标目标可能被误当背景；这不是完全无标注学习。验证集和测试集不参与参数更新，不需要审核 crop 或引入外部分类数据集。

## 结果与继续训练

默认输出：`runs/wheat_fourclass_joint/fourclass_train/`，同名目录已存在时自动递增。里面包括 `weights/best.pt`、`weights/last.pt`、`results.csv`、训练曲线、检测混淆矩阵，以及 `region_sampling.json`。日志中的 `region_loss` 已乘以区域损失权重。

`region_sampling.json` 按五类记录 GT、候选、实际选择的区域数量。计数包含增强和重复区域，只统计本次进程，不代表独立图片数；续训会重新计数。训练按审核后的检测 mAP 选择 best.pt。

只对尚未完成、保有优化器状态的 last.pt 断点续训：

```powershell
python -m my_scripts.train_yolov8 train --model runs/wheat_fourclass_joint/fourclass_train/weights/last.pt --data E:/Dataset/dataset_four_classes/data.yaml --resume
```

已正常完成的权重继续微调时，省略 `--resume`。从 Git 获取本项目时，将 YOLOv8n 预训练权重放到 `weights/yolov8n.pt`；项目包已包含它。迁移新环境需要先安装匹配的 PyTorch／torchvision，并在根目录运行 `python -m pip install -e .`。

## 验证和预测

运行独立验证，生成四类 Precision、Recall、F1、mAP 的 `per_class_metrics.csv` 和检测混淆矩阵：

```powershell
python -m my_scripts.train_yolov8 val --model runs/wheat_fourclass_joint/fourclass_train/weights/best.pt --data E:/Dataset/dataset_four_classes/data.yaml --name review_val
```

关闭审核做对照：

```powershell
python -m my_scripts.train_yolov8 val --model runs/wheat_fourclass_joint/fourclass_train/weights/best.pt --data E:/Dataset/dataset_four_classes/data.yaml --review-threshold 0 --name no_review_val
```

在验证集比较审核阈值时保持其他参数一致，例如分别设为 0.20、0.35、0.50，每次更换 `--name`。这里的 P／R／F1 来自框架的置信度曲线工作点，不能当作固定检测置信度 0.25 下的计数；比较时也检查完整 PR 曲线与召回损失。检测混淆矩阵含背景行列，这是框架对误报和漏报的统计，不表示新增了第五类检测输出。

如需判断辅助训练本身是否有帮助，另用相同设置、相同预训练权重训练一次 `--region-weight 0 --name detector_only_seed42`。该设置关闭辅助学习和审核；仅在同一权重上关闭审核，不能替代这个训练对照。三层二分类与本项目对比时也需保持轮数、数据、增强、种子等设置一致。

阈值只在 val 选择，确定后再用 `--split test` 做最终评估，不根据 test 反复调参。单次短训练只验证流程，不能证明精度提高。联合训练会改变共享特征，四类指标都应检查。

预测并保存标注图：

```powershell
python -m my_scripts.train_yolov8 predict --model runs/wheat_fourclass_joint/fourclass_train/weights/best.pt --source E:/Dataset/dataset_four_classes/images/test --name review_predict
```

请使用上述自定义入口。普通 `yolo predict`／`yolo val` 不会自动启用审核。迁移权重时需保留 `my_scripts/fourclass_joint.py`，因为检查点包含这个模型类。当前验证范围为 YOLOv8 PyTorch 单 GPU／CPU，部署导出和多 GPU 不在此实验范围内。ROIAlign 的 CUDA 反向可能提示非完全确定性，固定种子不能保证每次结果逐位相同。

## 实现位置

`my_scripts/fourclass_joint.py` 复用了现有二分类联合实验的模型、Trainer／Validator／Predictor 扩展方式，改为五类交叉熵、全类别候选取样、三层卷积及四类审核。原主分支尚无区域训练 owner，因此将必要行为集中在这个实验模块；不复制框架训练循环，不改动核心库，不新增依赖。代码超过 50 行的原因是训练、重载、续训和推理审核必须采用一致的模型与判定规则。根目录其他历史 PlantDoc 脚本不用于本实验。

## 已完成的验证

2026-10-21 在本机 Python 3.11／PyTorch 2.13 的 CPU 上，用真实数据子集（8 张训练、4 张验证）完成两轮短训练，中途保存后实际断点恢复。另检查了五类自动取样、三层卷积与共享骨干的非零梯度、原检测损失不变、无 GT 图片、空检测、模型重载，以及保留框逐项不变和背景拒绝四类框的行为。验证入口已实际导出四类 CSV 和混淆矩阵，一键启动器的参数入口也已检查。Ruff 格式及 F/I/D/UP/RUF/FA 检查通过。

这些验证确认流程和判定规则可运行；尚未进行本版本的完整训练、GPU 训练或独立测试集精度评估。
