# 智能分拣视觉：最小运行版本

这个分支先针对一台 USB 摄像头和没有机械臂下位机的情况，验证视觉主链路：

```text
相机预览 -> YOLO 形状检测 -> 颜色粗分类 -> 托盘标定 -> 输出 X/Y/Z
```

三相机联动、双目测高、真实机械臂通信不在这个版本中。它们会在单相机链路稳定后接入。

## 安装

在项目目录中创建或使用虚拟环境：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

把 `yolo11n.pt` 放在项目根目录，或者通过 `--weights` 指定权重。没有权重时程序仍可用于预览、拍照和标定。先探测相机编号：

```powershell
.\.venv\Scripts\python.exe camera_probe.py
```

## 运行

```powershell
.\.venv\Scripts\python.exe mvp_single_camera.py --camera 0 --weights .\yolo11n.pt
```

常用按键：

- `c`：冻结当前画面，依次点击托盘左上、右上、右下、左下四个角，按 `Enter` 保存标定；
- `s`：保存当前原图；
- `p`：保存当前识别结果 JSON；
- `q` 或 `Esc`：退出。

没有相机时也可以先运行逻辑自检：

```powershell
.\.venv\Scripts\python.exe mvp_single_camera.py --self-test
```

处理一张已经拍好的图片，不打开摄像头：

```powershell
.\.venv\Scripts\python.exe mvp_single_camera.py --image .\captures\test.jpg --headless --weights .\yolo11n.pt
```

标定结果保存在 `config/tray_calibration.json`，识别结果保存在 `reports/latest_detections.json`。标定文件被 `.gitignore` 忽略，因为它属于现场设备参数。

## 当前输出

每个目标会输出：

```json
{
  "shape": "0",
  "color": "red",
  "x_mm": 120.5,
  "y_mm": 85.2,
  "z_mm": null,
  "confidence": 0.91
}
```

在 `config/objects.json` 中填写实际物块高度后，程序会按形状输出 `z_mm`。机械臂接口目前使用模拟输出。

例如：

```json
{
  "objects": {
    "cube": {"shape": "cube", "height_mm": 30}
  }
}
```

