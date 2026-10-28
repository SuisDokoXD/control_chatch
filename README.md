# 智能分拣视觉

仓库当前有两条视觉链路：老的 YOLO/MVP 单相机链路（`程序/mvp_single_camera.py`，仍是 MVP），以及纯 OpenCV 双视角几何链路（`scripts/two_view_shape.py` + `scripts/two_view_model_fit.py`）：俯视定平面位置、侧视定高度，两路完整轮廓一起拟合直立规则实体，输出托盘毫米坐标、包络尺寸、高度与抓取候选。模型假设直立、完整、无遮挡；拟合不通过就拒判并给出原因，不硬分类。

`程序/three_camera_capture.py` 支持配置垂直相机与两台侧面相机、采集帧组、预览和保存，但它只做采集，不做跨视角融合；三路设备编号需配置。采集采用依次 `grab` 再 `retrieve`，只提供软件调用时间差诊断，不等于硬件同步。整机方案不使用 ToF。机械臂接口与计数状态机仍未连接实际下位机，`robot_target_ready` / `robot_command_ready` 保持 false。

**换到另一台电脑**：见 [文档/迁移到新电脑.md](文档/迁移到新电脑.md)。仓库只带代码、配置、测试与文档；相机编号、标定、空盘参照帧都必须在新机器上重做。

## 拍照入口

在项目根目录执行：

```powershell
.\启动数据集采集.ps1 --classes square
```

也可以双击 `启动数据集采集.cmd`，默认使用 square 图片名前缀。两个启动脚本都支持传入 --classes、--camera、--output，不传参数时默认采集 square。

```powershell
.\启动数据集采集.ps1 --classes square,mixed_sparse,mixed_dense,difficult,background
```

数字键 1～5 切换上述前缀，空格拍照，Q/Esc 退出。前缀只用于图片命名，不会自动设置 YOLO 类别或生成标签。

图片保存在 `dataset/yolo_cube/images/raw/`，例如 square_0001.jpg；LabelImg 的 YOLO 矩形框标签放在 `dataset/yolo_cube/labels/raw/`，同名图片与标签对应。原图保留彩色。

## 三相机采集入口

先在 `配置/camera_roles.json` 为 `vertical`、`side_a`、`side_b` 配置互不相同的设备编号。两台侧面相机当前编号留空，请先运行 `py -3.13 .\程序\camera_probe.py --max-index 9` 确认编号再填写。也可以用对应的 `--vertical-camera`、`--side-a-camera`、`--side-b-camera` 参数临时覆盖。

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) '.venv\Lib\site-packages'
py -3.13 .\程序\three_camera_capture.py
# 配好三个编号后，无窗口采集并保存 5 组三视角图像
py -3.13 .\程序\three_camera_capture.py --headless --frames 5
# 不连相机的合成图自检
py -3.13 .\程序\three_camera_capture.py --self-test
```

预览窗口按 S 保存当前三路图像，Q/Esc 退出；无窗口模式会自动保存每组图像和 `metadata.json` 到 `captures/three_camera/`。采集按顺序调用三路 `grab()` 再 `retrieve()`，记录的是软件调用时间差，**不代表硬件同步或曝光时间一致**；本入口尚不做标定、跨视角匹配/融合或高度估计。

## OpenCV 形状识别（实验工具）

`程序/opencv_shape_vision.py` 是不依赖训练模型的形状识别入口：用托盘背景建模 + 色度/暗度门限
分出物块，用距离变换种子 + 颜色梯度分水岭拆开粘连，再按轮廓几何和内部明暗结构判定 9 类形状
（颜色只用于把物块和托盘分开，不参与形状判定）。托盘内部才是工作区，托盘边沿、左侧台面和
画面黑边会先被 ROI 排除。

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) '.venv\Lib\site-packages'
py -3.13 .\程序\opencv_shape_vision.py --self-test
py -3.13 .\程序\opencv_shape_vision.py --eval-labels
py -3.13 .\程序\opencv_shape_vision.py --limit 40 --save-images --json captures\shape_cv\results_40.json
py -3.13 .\scripts\make_shape_report.py --json captures\shape_cv\results_40.json
```

标注图与 JSON 输出到 `captures/shape_cv/`，`review.html` 是可直接翻阅的复查页面。
工作区由**托盘四角的四个黑点**定位（不是靠"最亮的低色度块"猜）：把检测到的角点对齐到
`TRAY_DOT_REF` 基准布局，再用单应/仿射推出完整托盘四边形，因此左侧灰色台面和画面黑边
不会进入工作区。找不到黑点时退回原来的启发式 ROI。
严格一对一评估（5 张已标注图、44 个标注框）：检测召回 **79.5%**、检测精确率 **64.8%**；
分类判对 **24** / 判错 **2** / 拒判 **9**，**已判定准确率 92.3%**，端到端正确率 **54.5%**。
453 张 raw 全量运行：检出 2927、空托盘 10 张零误检、**非空图无零检出**、
37.7% 检出被拒判、14.5% 检出成为抓取候选。
掩膜缺角、形状在单视角下不可分的块都输出 `unknown` 而不是硬分类，这是"宁可不抓也不抓错"的设计。
想换覆盖率可以关掉对应歧义规则，例如
`--set tri_ambig_band=0.0 --set prism_ambig=0 --set disc_ambig=0`。
**注意**：检测指标仍受标注质量限制——44 个框里有 1 个重复框，`awayfixed_0035`
只标了 2 个框却画面里约 10 块。
注意 `--eval-labels` 是宽松口径（不惩罚多余预测），数字会偏高，对比时以
`scripts/evaluate_shape_vision.py` 为准。方法、混淆矩阵、失败原因和后续建议见
[文档/OpenCV形状识别实验报告.md](文档/OpenCV形状识别实验报告.md)。该工具不训练模型、不做托盘毫米
坐标换算，也不替代 `程序/mvp_single_camera.py` 的 MVP 链路。

工具同时给出**抓取顺序**：对每个物块算净空（与邻块、托盘边沿的空隙）、形状歧义度
（是否落在已知混淆区）和"是否摆在白托盘上"，三项都通过才标为可抓，再按
0.55×净空 + 0.30×置信度 + 0.15×规整度排序；不可抓的块写明原因，仍然输出。
标注图上 `#1`、`#2` 是推荐顺序，`x` 表示不建议抓。相关参数：
`--clearance-ratio`、`--min-clearance-px`、`--max-ambiguity`；
任何阈值都可以用 `--set KEY=VALUE` 覆盖（可重复，键必须是 `DEFAULTS` 里已有的项）。

## 环境与其他工具

当前保留相机排查中验证过的启动兼容方案：通过 py -3.13 启动普通 Python，同时加载项目 .venv/Lib/site-packages 依赖。直接使用 .venv/Scripts/python.exe 的摄像头问题尚未彻底修复；此兼容方式不代表已完成独立环境重建。

其他工具可以在项目根目录的 PowerShell 中这样运行：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) '.venv\Lib\site-packages'
py -3.13 .\程序\camera_probe.py --max-index 3
py -3.13 .\程序\mvp_single_camera.py --self-test
py -3.13 .\程序\mvp_single_camera.py --role vertical
```

`配置/camera_roles.json` 保存垂直相机和两台逻辑侧相机（`side_a`、`side_b`）编号；侧相机编号需实际填写。`camera_settings.json` 中的尺寸、帧率和编码用于三路采集；曝光、增益、白平衡仅在明确传入 `--apply-camera-settings` 时应用。旧单相机 MVP 的托盘标定 `配置/tray_calibration.json` 只对应垂直视角，不等同于三相机标定。

## 辅助脚本

scripts/ 保留一套工具，旧的编号目录及重复副本已移出工作区。

| 文件 | 用途 / 默认位置 |
| --- | --- |
| record_camera.py | 录像到 dataset/videos/ |
| extract_dataset_frames.py | 抽帧到 dataset/extracted/；彩色及灰度副本分开 |
| make_shape_report.py | 把 opencv_shape_vision.py 的 JSON 结果整理成 HTML 复查页面 |
| evaluate_shape_vision.py | 一对一 IoU 评估：检测召回/精确率、漏检、误检、分类准确率 |
| auto_label.py | 指定已训练模型进行预标注，输出到 dataset/auto_labels/ |
| calibrate_tray.py | 独立 YAML 标定工具；输出到 配置/扩展/calibration.yaml |
| train_cpu.py | 读取 dataset/yolo_cube/data.yaml 和 权重/yolo11n.pt，输出到 runs/ |
| prepare_yolo_input.py | 可选灰度副本转换，原图不修改 |

MVP 的 JSON 标定和独立工具的 YAML 标定格式不同，不可直接互换。扩展模块与 YAML 模板目前未接入 MVP。

训练前先完成标注和 train/val/test 划分，并核对 data.yaml 的类别顺序；当前仍为 cube 示例，不代表 square 或十类数据集已经配置完成。通用 yolo11n.pt 尚未训练成物块模型。

目录职责见 [项目目录说明.md](项目目录说明.md)。当前架构以 [文档/智能分拣视觉方案阶段性总结.md](文档/智能分拣视觉方案阶段性总结.md) 为准；同名 Word 文档尚未同步为当前架构。历史资料中的旧命令以本说明为准。

