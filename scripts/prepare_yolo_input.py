"""Create grayscale 3-channel copies while preserving color originals."""
import argparse
from pathlib import Path
import cv2

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    count = 0
    for src in a.source.rglob('*'):
        if src.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp','.webp'}: continue
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None: continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dst = a.output / src.relative_to(a.source)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        count += 1
    print(f'converted={count}')

if __name__ == '__main__': main()
