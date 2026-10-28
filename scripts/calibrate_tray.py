"""Manual four-corner tray calibration; works with a camera or a still image."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]

points = []
def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4: points.append((x, y))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='0', help='camera index or image path')
    p.add_argument('--width-mm', type=float, required=True)
    p.add_argument('--height-mm', type=float, required=True)
    p.add_argument('--output', type=Path, default=PROJECT_DIR / '配置' / '扩展' / 'calibration.yaml')
    a = p.parse_args()
    cap = None
    if Path(a.source).exists(): frame = cv2.imread(a.source)
    else:
        cap = cv2.VideoCapture(int(a.source)); ok, frame = cap.read()
        if not ok: raise RuntimeError('cannot read camera')
    cv2.namedWindow('tray calibration'); cv2.setMouseCallback('tray calibration', click)
    while True:
        display = frame.copy()
        for i, pt in enumerate(points): cv2.circle(display, pt, 6, (0,255,0), -1); cv2.putText(display, str(i+1), pt, 0, 1, (0,255,0), 2)
        cv2.imshow('tray calibration', display)
        key = cv2.waitKey(30) & 0xff
        if key == 27: break
        if key in (ord('r'), ord('R')): points.clear()
        if key in (ord('s'), ord('S')) and len(points) == 4: break
        if cap is not None and not points:
            ok, frame = cap.read()
            if not ok: break
    cv2.destroyAllWindows()
    if len(points) != 4: raise RuntimeError('need four corners; click TL, TR, BR, BL then press S')
    src = np.float32(points); dst = np.float32([[0,0],[a.width_mm,0],[a.width_mm,a.height_mm],[0,a.height_mm]])
    h = cv2.getPerspectiveTransform(src, dst)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(yaml.safe_dump({'image_size':[int(frame.shape[1]),int(frame.shape[0])], 'tray_size_mm':[a.width_mm,a.height_mm], 'corners_px':points, 'homography_px_to_mm':h.tolist(), 'robot_transform':None}, sort_keys=False), encoding='utf-8')
    print(f'saved={a.output}')

if __name__ == '__main__': main()
