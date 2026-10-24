# 冻结原 400 轮 YOLO 的两个复核实验

同一份四类 YOLOv8 检测器保持固定，分别训练白粉病二分类复核器、四分类复核器。检测器的骨干、颈部、检测头、BatchNorm 参数和运行统计量均不更新。两个版本使用同一个实现、同一份初始检测权重、数据划分和训练配置，分别保存结果。正式训练由使用者启动。

## 两个版本

| 项目     | `binary` 白粉病二分类      | `fourclass` 四分类                               |
| -------- | -------------------------- | ------------------------------------------------ |
| 分类答案 | 白粉病／非白粉病           | 褐锈病、健康、散黑穗病、白粉病                   |
| 输出层   | 1 个值，Sigmoid            | 4 个值，Softmax                                  |
| 训练损失 | 正负组各自平均后平均的 BCE | 各出现类别平均后平均的交叉熵                     |
| 复核范围 | 仅白粉病检测框             | 全部四类检测框                                   |
| 拒绝规则 | 白粉病概率低于阈值         | 最高概率类别与原类别不一致，或原类别概率低于阈值 |

两个版本都只筛框，不改类别、坐标或检测置信度，不新增背景类，也不补回漏检框。关闭复核时四类检测输出应与原基线一致。复核可能损失召回率，是否改善最终检测精度须以完整评估为准。

共同结构：

```text
图像（训练时先做增强） → 固定 YOLOv8 → 原四类检测头 → NMS → 复核保留／删除
                          └ P3 特征（detach）
                            → ROIAlign 7×7，每侧扩边 10%
                            → 3×3 Conv64 + ReLU
                            → 3×3 Conv64 + ReLU
                            → 3×3 Conv64 + ReLU
                            → 全局平均池化 → 全连接（1 或 4 个输出）
```

优化器只接收复核器参数；检测器始终使用 eval 模式和 no_grad 前向。训练循环、EMA、调度、保存和数据增强复用项目原有框架。

## 已确认的默认参数

配置文件为根目录 `frozen_review.yaml`，两版共用。

| 参数                        | 数值                               |
| --------------------------- | ---------------------------------- |
| 训练轮数／早停              | 各 30 轮，`patience=0` 关闭早停    |
| batch／输入尺寸／workers    | 8／640／0                          |
| 优化器                      | AdamW，beta1=0.9、beta2=0.999      |
| 初始学习率／末端比例        | 0.001／0.01，线性调度              |
| weight decay                | 0.0001，bias 不衰减                |
| warmup                      | 1 轮，bias 初始学习率 0            |
| 梯度累计                    | `nbs=batch`，每个 batch 更新一次   |
| seed／确定性设置            | 42／开启框架 deterministic 设置    |
| AMP／channels_last／compile | 均关闭                             |
| 设备                        | 单 GPU 0；可用 `--device cpu`      |
| 最佳权重                    | 按验证集复核分类损失选择 `best.pt` |

增强参数明确沿用现有 `wheat_fourclass_joint/runs/fresh_400/seed0/args.yaml`：

| 增强                                 | 数值                |
| ------------------------------------ | ------------------- |
| Mosaic                               | 1.0，最后 10 轮关闭 |
| 水平翻转                             | 0.5                 |
| 平移／缩放                           | 0.1／0.5            |
| HSV 色相／饱和度／亮度               | 0.015／0.7／0.4     |
| 旋转／剪切／透视／垂直翻转／BGR 交换 | 全部为 0            |
| MixUp／CutMix／CopyPaste             | 全部为 0            |

沿用原检测增强管线中的可选 Albumentations：环境安装该包时使用原默认变换，未安装时原框架跳过。新实验共用同一环境，不新增依赖；若跨机器比较，应同时固定该包的安装状态和版本。`auto_augment`、`erasing` 是图像分类任务的参数，不用于本检测复核流程。训练配置中的 `augment=false` 表示不开启验证／预测的测试时增强，并不关闭上述训练增强。ROI 特征本身没有另外的随机增强。

30 轮训练的最后 10 轮为第 21–30 轮。旧项目 400 轮的最后 10 轮为第 391–400 轮：关闭时机参数相同，所占比例不同。

## 自动生成区域监督

继续使用 `E:/Dataset/dataset_four_classes/data.yaml` 的原 train／val／test 划分，不需要重新人工标注。

1. 两个版本都从全部四类 GT 框和固定检测器的 NMS 候选生成区域。候选阈值 `conf=0.1`、NMS IoU=0.7、max_det=300。
2. GT 框按真实类别监督；候选框在扩边前与 GT 的 IoU≥0.5 时，按匹配 GT 的真实类别监督。YOLO 的预测类别不作为训练答案。
3. 扩边区域若与其他类别 GT 相交，则视为含糊区域并跳过。未匹配候选直接跳过，不自动标为背景。
4. 每张图片、每个原始类别最多选 32 个区域。两版共享这套按原始四类取样的规则，使用独立且同种子的区域随机生成器，避免输出层大小影响取样随机序列。
5. 二分类在区域选定后，将真实类 3 映射为正样本、其余三类映射为负样本。四分类保留原类别。二分类正负组平衡、四分类按类别平衡，损失数值不能跨任务直接比较。

`region_sampling.json` 记录按原四类统计的 GT、候选和实际选择数量，以及含糊／未匹配跳过数。数量包含增强与重复采样，并非独立图片数。漏标或错标仍会影响监督；这两版都没有可靠背景负样本监督，四分类低置信度不能视作经过训练的背景判断。

## 本地启动

在 PyCharm 打开本项目文件夹。选择已有解释器 `E:\Programes\ANACONDA\envs\pytorch311\python.exe`，无需升级环境。两个启动器优先使用这个解释器，其他机器使用当前环境的 `python`。

本地交付包含 `weights/baseline400.pt`，来自原项目的 `runs/wheat/v8n_four_classes_200epochs-4/weights/best.pt`。原文件名里的 200 不作为训练轮数依据。本次复用的文件 SHA256 为：

```text
c161ee4819256b2009e1fe3edabc3b839ebd5049881fbd2f0658e0aec22320c5
```

Git 仓库不提交数据集或权重。从仓库获取代码后，应自行放入同一份基线权重，或传入 `--baseline` 指定路径；使用已有匹配的 PyTorch／torchvision 环境。

两个版本分别运行，建议依次启动：

```powershell
.\train_binary.bat
.\train_fourclass.bat
```

命令可追加覆盖参数，例如 `--batch 4`、`--seed 0`。做对比时请对两个版本应用相同覆盖。两个项目不会相互读取已训练的分类器，均从相同基线检测器和随机初始化的分类器开始，不会将四分类接着二分类继续训练。

默认输出分别为：

```text
runs/frozen_review/binary_seed42/weights/best.pt
runs/frozen_review/fourclass_seed42/weights/best.pt
```

重复运行会递增目录编号，请使用终端实际打印的目录。每次输出包含 `best.pt`、`last.pt`、`args.yaml`、`results.csv`、`region_sampling.json` 和训练／验证图；成功完成后另写 `experiment.json`，包含基线文件哈希、检测器张量哈希、数据 YAML 哈希、PyTorch 版本和本次配置。每轮验证前检查当前模型与 EMA 检测器全部持久张量，保存后再次检查基线一致性。

训练日志的检测指标关闭复核，便于核对基线；分类器学习情况看 `review_loss`。复核效果需用下面的独立验证命令评估。

## 统一比较和最终测试

先在 PyCharm 终端指定正确的解释器：

```powershell
$frozenPython = 'E:\Programes\ANACONDA\envs\pytorch311\python.exe'
```

两个版本训练完成后，一次比较基线及两个版本的多个复核阈值：

```powershell
& $frozenPython -m my_scripts.train_frozen compare --model runs/frozen_review/binary_seed42/weights/best.pt runs/frozen_review/fourclass_seed42/weights/best.pt
```

默认阈值为 0、0.1、0.15、0.2、0.25、0.5，0 完全关闭复核。比较命令固定检测 `conf=0.001`、NMS IoU=0.7、max_det=300、FP32，汇总保存在 `outputs/frozen_comparison/metrics.csv`，重复比较会递增目录。关闭复核的两版指标应分别与基线一致，否则比较中止。

CSV 包含各类和宏平均的 Precision、Recall、F1、AP50、AP50-95。P／R／F1 来自框架选择的置信度曲线工作点，不表示固定部署置信度 0.25 时的计数；无 GT 的类别没有可估计 AP，因此该类不生成行。二分类阈值和 Softmax 阈值含义不同，应各自在相同 val 上选工作点，再比较召回损失和最终检测指标，不能只比较同一个数字阈值。

对一个版本验证并生成混淆矩阵、PR 曲线、预测示例和 `per_class_metrics.csv`：

```powershell
& $frozenPython -m my_scripts.train_frozen val --model runs/frozen_review/fourclass_seed42/weights/best.pt --review-threshold 0.25
```

`0.25` 是示例，未经过完整验证集调优。确定阈值后，在独立 test 上只评估选定工作点：

```powershell
& $frozenPython -m my_scripts.train_frozen val --model runs/frozen_review/fourclass_seed42/weights/best.pt --review-threshold 0.25 --split test
```

阈值扫描限定在 val；不要使用 test 反复调参。检查保存的检测器：

```powershell
& $frozenPython -m my_scripts.train_frozen check --model runs/frozen_review/fourclass_seed42/weights/best.pt
```

预测并保存标注图片，检测置信度使用 0.25，复核阈值独立设置：

```powershell
& $frozenPython -m my_scripts.train_frozen predict --model runs/frozen_review/fourclass_seed42/weights/best.pt --source E:/Dataset/dataset_four_classes/images/test --review-threshold 0.25
```

请使用这个自定义入口；普通 `yolo val`／`yolo predict` 不会自动执行复核。迁移权重时保留 `my_scripts/frozen_review.py` 和本项目源码。当前范围为 YOLOv8 原生 PyTorch、CPU／单 GPU，不包含导出模型的复核分支，也不提供断点续训入口。

## 对消融结果的解释

本项目两个冻结版本的分类器主体、区域取样、数据增强和训练预算一致；任务标签、输出层和相应分类损失按任务不同。这可以比较两种复核方案。关闭复核与原基线的对照用于验证固定检测器并衡量筛框的效果。

已有 `wheat_fourclass_joint` 是联合训练、五类区域监督（含背景），且原训练为 400 轮、batch=4、seed=0、optimizer=auto、AMP 开启。它与本项目还存在冻结方式、背景监督和训练预算等差异，不能作为仅改变冻结方式的严格单变量消融。需要这种消融时，另行对齐这些因素并使用相同数据划分与验证设置。单次 seed 的结果也不反映多次重复实验的波动。

## 已完成的流程验证

在 8 张真实训练图片、4 张真实验证图片上，两个版本分别完成 2 轮 GPU 训练，batch=8、imgsz=640、AMP 关闭、随机增强开启。小样本训练临时设 `close_mosaic=0`，以覆盖拼图路径；正式配置仍是最后 10 轮关闭，并另外验证了关闭 Mosaic 后的数据加载。

已确认：四类都有区域被选中；两个任务的区域取样统计一致；检测器与 EMA 的全部持久张量不变；保存重载后原始检测输出和 P3 特征与基线逐项一致；三层卷积都有梯度且梯度不进入检测器或输入特征；无标注、空检测和阈值 0／1 的筛选范围正确。

CPU 上已实际执行两个版本的验证、预测及阈值 0、0.25、0.5 的统一比较，导出 35 行指标（7 组×四类及总体），关闭复核的全部汇总指标与基线一致。Ruff 格式及 F／I／D／UP／RUF／FA 检查通过。本地交付的 `verification/checks.json` 和日志保存验证记录；小样本权重单独标识，不作为正式训练结果。

上述验证确认代码流程可运行；两个版本均未启动正式 30 轮训练，尚无完整数据集精度增益结论。
