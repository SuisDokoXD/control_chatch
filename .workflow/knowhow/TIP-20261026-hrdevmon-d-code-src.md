---
id: "TIP-20261026-hrdevmon-d-code-src"
title: "火绒 hrdevmon 拦掉了 D:\\code\\src 下解释器的摄像头访问"
keywords: ["火绒", "hrdevmon", "摄像头保护", "camera", "VideoCapture", "DSHOW", "can't be used to capture by index", "相机打不开", "PYTHONPATH", "venv"]
status: active
created: 2026-10-26
source: pi-desktop
---

症状：项目 .venv\Scripts\python.exe 跑任何 cv2.VideoCapture（DSHOW/MSMF 都试）都失败，日志 `cap.cpp:480 VIDEOIO(DSHOW): backend is generally available but can't be used to capture by index`，连内置相机都开不了；但 ffmpeg 能正常抓帧。
排查结论：cv2 本身没问题——venv 与系统 Python 的 cv2.pyd/config.py 哈希一致，用系统解释器加载 venv 的 cv2 也正常；把同一份 venv 复制到不同位置测试，只有路径前缀 D:\code\src\ 下的解释器被拦（D:\code\venv_probe、D:\venv_probe、D:\code\srcx\v、D:\codes\src\v 全部正常）。拦截者是火绒：Camera/Image 设备类 UpperFilters = hrdevmon,ksthunk，hrdevmon.sys 运行中。
规避：相机脚本用系统解释器 C:\Users\123\AppData\Local\Programs\Python\Python313\python.exe，并设 $env:PYTHONPATH="D:\code\src\sight\.venv\Lib\site-packages"（scripts/camera_preview.py 和 scripts/check_board_visible.py 的文档串里已写明）。
根治：在火绒「访问控制 → 摄像头保护」里放行 D:\code\src\sight\.venv\Scripts\python.exe（或整个 D:\code\src 目录）。
