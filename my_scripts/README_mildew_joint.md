# 白粉病三层卷积分支：独立联合训练项目

本实验直接复用四分类检测数据集的图片和 YOLO txt 标注，自动生成白粉病二分类监督。无需人工审核 crop、无需 review.csv、无需额外的 WFD 数据集或原 ResNet18 权重。当前提供单 GPU / CPU 的 PyTorch 训练、验证和图片预测。

## 学习方式

YOLOv8 保留原四分类检测分支，新增共享 P3 特征的区域二分类分支：ROIAlign 取 7×7 区域特征，再经过三层 3×3 卷积（每层 64 通道、ReLU）、全局平均池化和一个全连接层输出白粉病概率。分类损失会回传到分类分支和共享检测特征；候选框坐标和自动标签不参与梯度计算。总损失为原检测损失加 `binary_weight × 二分类损失`，默认权重 0.5。两个分支保存在同一份 best.pt / last.pt 中。

正负样本每次直接在增强后的训练图片上自动构建：

- 正样本：已有的白粉病 GT 框，以及扩边后与白粉病 GT 的 IoU≥0.50 的检测候选。
- 负样本：已有其他三类的 GT 区域，以及与其他类 GT 的 IoU≥0.50 的候选；这些区域还必须与所有白粉病 GT 没有交集。
- 背景负样本：与所有 GT 都没有交集的候选区域。
- 其余模糊候选自动跳过，不需要逐张人工判断。

每张图取白粉病检测分数最高的至多 64 个候选；每侧扩边 10% 后再判定重叠，最多抽取 32 个正样本和 32 个负样本，按当前批次中存在的正负类别分别平均 BCE。增强、检测匹配、优化器、EMA 和保存流程复用原训练器。`binary_sampling.json` 记录实际样本来源与数量；这些是包含重复、增强区域的计数。

自动监督依赖原检测标注。无交集只表示“标注中无目标”，漏标仍可能造成错误负样本；这里不是完全无标注学习。验证、测试标签不参与参数更新。每轮 val 可以计算损失与指标，但没有反向传播；测试集只用于最终评估。

预测时先使用标准类别独立 NMS，再只对 Mildew 框计算二分类分数并按阈值过滤。其他类不经过二分类过滤。**联合训练会改变共享特征，所以其他类的训练后指标也可能变化**；“其他框不被过滤”不代表“指标必然不变”。

## 本地运行

在 PyTorch / torchvision 匹配的 Python 环境中，进入本项目包根目录。建议沿用你已训练成功的 Python 3.11 CUDA 环境。

```powershell
python -m pip install -e .
python -m my_scripts.train_yolov8 train --model weights/yolov8n.pt --data E:/Dataset/dataset_four_classes/data.yaml --epochs 100 --batch 2 --imgsz 640 --device 0 --name joint_3conv_seed42
```

包内 `weights/yolov8n.pt` 是原 YOLOv8n 预训练权重，新二分类分支从随机初始化开始，与检测器一起学习。无需使用旧 ResNet18。若直接从 Git 仓库获取代码，请自行将预训练权重复制到上述位置。

本地默认 batch=2、workers=0；显存不足先将 batch 改为 1，640 输入先保持不变。CPU 使用 `--device cpu`，速度更慢。本项目不自动启动长时间完整训练。短训练只验证流程，不是可用于报告最终精度的模型。首次训练的 AMP 检查可能下载 yolo26n.pt，仅用于框架 AMP 自检，不用于初始化此检测器。

默认输出为 `runs/wheat_joint_3conv/joint_3conv_seed42`，目录已存在时框架会自动递增。日志出现 `mildew_loss` 即为加权后的二分类损失；训练结束应有 `weights/best.pt`、`weights/last.pt`、`results.csv` 和 `binary_sampling.json`。如需继续因中断而尚未结束的训练：

```powershell
python -m my_scripts.train_yolov8 train --model runs/wheat_joint_3conv/joint_3conv_seed42/weights/last.pt --data E:/Dataset/dataset_four_classes/data.yaml --resume
```

正常完成训练的 last.pt 会被框架移除优化器状态，继续微调这种权重时应省略 `--resume`。从联合权重重新开始一次微调也会加载其区域分类分支。

固定随机种子不保证 CUDA ROIAlign 反向传播完全确定；框架可能输出 warn_only 的确定性提醒。比较实验应保持种子及所有训练设置一致，正式结论建议重复多次训练。

## 四组阈值对照

训练结束后，用已经存在的阈值评估脚本检查 0.15、0.20、0.30、0.35，其他三类固定 0.25。新增 `joint` 类型直接读取检测模型内的二分类分支，不需要 `--classifier`：

```powershell
python -m my_scripts.mildew_refinement evaluate --model runs/wheat_joint_3conv/joint_3conv_seed42/weights/best.pt --classifier-type joint --data E:/Dataset/dataset_four_classes/data.yaml --thresholds 0.15 0.20 0.30 0.35 --cls-thresholds 0.50 --device 0 --split val --output outputs/joint_3conv_thresholds_val
```

输出九个工作点：四个检测阈值各自有/无过滤，加一个 0.25 无过滤基线。所有组共享同一份已经联合训练的权重，逐图检查非白粉病输出完全一致，结果在 `summary.csv`。`--output` 必须是尚不存在的目录。这里的“无过滤”仍使用联合训练后的检测器，不能替代单独训练原 YOLO 的对照。

为判断联合训练本身是否有价值，用相同预训练权重、训练设置与数据划分，再运行一次 `--binary-weight 0 --name detector_only_seed42`。权重为 0 时不学习二分类任务，也不启用该分支过滤。比较两次完整训练后四类的指标，才能区分额外训练和二分类监督的影响；不能把旧的某次训练截图直接当成等条件对照。

只在 val 选取检测和二分类阈值。确定后再以 `--split test` 运行，换一个输出目录，不根据 test 反复调参。背景减少的同时必须检查白粉病召回和 F1，当前代码不保证精度一定提升。

## 标准 mAP 验证和图片预测

下面 val 使用框架常规的低检测置信度计算 AP，二分类阈值默认 0.50；口径与前面的固定阈值 summary.csv 不同。训练过程使用同一验证器，并按过滤后的检测指标选择最佳检查点。

```powershell
python -m my_scripts.train_yolov8 val --model runs/wheat_joint_3conv/joint_3conv_seed42/weights/best.pt --data E:/Dataset/dataset_four_classes/data.yaml --device 0 --binary-threshold 0.50 --name joint_3conv_val

python -m my_scripts.train_yolov8 predict --model runs/wheat_joint_3conv/joint_3conv_seed42/weights/best.pt --source E:/Dataset/dataset_four_classes/images/test --device 0 --binary-threshold 0.50 --name joint_3conv_predict
```

`--binary-threshold 0` 关闭过滤。普通 `yolo predict` / `yolo val` 可以读取包含此模块的检测权重，但不会自动启用自定义过滤器；请使用上述命令。保留项目中的 `my_scripts/mildew_joint.py`，不要单独复制权重到不含该模块的环境。当前实验只覆盖四类 YOLOv8 的 PyTorch 单卡流程，未实现部署导出或多 GPU 分布式训练。

## 独立项目与一层版本的对照

此项目位于独立的 `wheat_mildew_joint_3conv` 文件夹，包含自己的代码、预训练权重、启动入口和结果目录，不引用一层项目的代码或输出。原一层项目保持原样。共同使用原四分类数据集，无需复制一份图片数据。

唯一的建模改动是将二分类头由一层卷积增加为三层卷积；P3 特征、ROIAlign 7×7、扩边、自动取样、池化方式、BCE、损失权重 0.5 和二分类阈值 0.50 均保持不变。YOLOv8n 的 64 通道 P3 输入下，该分类头有 110,849 个参数，一层版为 36,993 个参数。完整训练仍需与一层版按相同数据、轮数、种子和阈值进行比较，不能预先断言三层更好。

新检查点使用独立的 `ThreeConvDetectionModel` 类；请使用本项目自己的代码加载三层版权重。为保证对照，训练初始化使用 `weights/yolov8n.pt`，不要使用一层版的联合权重初始化此实验。

在 PyCharm 中打开整个 `wheat_mildew_joint_3conv` 文件夹，解释器选择 `E:/Programes/ANACONDA/envs/pytorch311/python.exe`。双击根目录 `start_local_training.bat`，或在项目根目录终端执行 `./start_local_training.bat`，启动 100 轮、640 输入、batch=2 的 GPU 训练。换电脑时修改启动脚本中的解释器和数据路径。

本次只在 CPU 上做流程验证，因为本机 GPU 已有任务。包内不包含完整训练好的三层检查点；完整训练尚未执行，GPU 新版本的训练耗时与显存占用尚未实测。原检测标注如有漏标，自动背景监督仍可能有误。
