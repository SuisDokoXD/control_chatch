---
id: "TIP-20261026-charuco-22-marker-8-4-32"
title: "这块 ChArUco 标定板是 22 个 marker（不是 8×4=32）"
keywords: ["CharucoBoard", "标定板", "marker 数", "getIds", "getObjPoints", "charuco_screen_244x170mm", "检出率"]
status: active
created: 2026-10-26
source: pi-desktop
---

captures/calibration/charuco_screen_244x170mm.json 描述的屏幕板是 9×5 格 / 25mm 格边长，但 marker 只放在黑格上，共 22 个（ids 0..21），棋盘内角点 32 个。
统计检出率要用 len(board.getIds())，不要用 (squaresX-1)*(squaresY-1)=32（会显示成 11/32 这种偏低的假象）。
board.getObjPoints() 返回每个 marker 的 4 个角点 (x, y, 0)，取值时按 3 列 reshape 后取前两列。
