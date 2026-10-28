"""把 opencv_shape_vision.py 的结果整理成可翻阅的 HTML 页面。

用法（项目根目录）：
    py -3.13 .\\scripts\\make_shape_report.py --json captures\\shape_cv\\results_40.json ^
        --images captures\\shape_cv --output captures\\shape_cv\\review.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成形状识别结果复查页面")
    parser.add_argument("--json", type=Path, required=True, help="opencv_shape_vision.py 输出的 JSON")
    parser.add_argument("--images", type=Path, default=ROOT / "captures" / "shape_cv",
                        help="标注图所在目录")
    parser.add_argument("--output", type=Path, default=ROOT / "captures" / "shape_cv" / "review.html")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    json_path = args.json if args.json.is_absolute() else ROOT / args.json
    images_dir = args.images if args.images.is_absolute() else ROOT / args.images
    output = args.output if args.output.is_absolute() else ROOT / args.output

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    totals: Counter[str] = Counter()
    cards = []
    for name, objects in payload.items():
        stem = Path(name).stem
        thumb = images_dir / f"{stem}_shapes.jpg"
        counts = Counter(obj["label"] for obj in objects)
        totals.update(counts)
        picks = [obj for obj in objects if obj.get("pickable")]
        rows = ""
        for obj in objects:
            rank = f"#{obj['rank']}" if obj.get("pickable") else "—"
            reasons = "、".join(obj.get("reasons") or [])
            rows += (
                f"<tr class=\"{'ok' if obj.get('pickable') else 'no'}\">"
                f"<td>{rank}</td><td>{html.escape(obj['label'])}</td>"
                f"<td>{obj['confidence']:.2f}</td>"
                f"<td>{(obj.get('clearance_ratio') or 0):.2f}r</td>"
                f"<td>{(obj.get('ambiguity') or 0):.2f}</td>"
                f"<td>{html.escape(reasons)}</td></tr>"
            )
        image_tag = (f'<img src="{html.escape(thumb.name)}" alt="{html.escape(stem)}">'
                     if thumb.exists() else '<p class="missing">（未找到标注图）</p>')
        cards.append(f"""
    <section class="card">
      <h2>{html.escape(name)} <span>{len(objects)} 个物块 · 建议抓取 {len(picks)} 个</span></h2>
      {image_tag}
      <p class="counts">{html.escape('  '.join(f'{k}×{v}' for k, v in sorted(counts.items())))}</p>
      <details open><summary>抓取顺序</summary>
        <table><thead><tr><th>优先级</th><th>形状</th><th>置信度</th><th>净空</th><th>歧义</th><th>原因</th></tr></thead>
        <tbody>{rows}</tbody></table>
      </details>
    </section>""")

    summary = "  ".join(f"{k}×{v}" for k, v in sorted(totals.items()))
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>OpenCV 形状识别与抓取排序复查</title>
<style>
  body {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; margin: 0; padding: 24px;
         background: #12161c; color: #e8eaed; }}
  h1 {{ font-size: 20px; margin: 0 0 8px; }}
  .summary {{ color: #9fb3c8; margin-bottom: 20px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }}
  .card {{ background: #1b2129; border: 1px solid #2a323d; border-radius: 10px; padding: 12px; }}
  .card h2 {{ font-size: 14px; margin: 0 0 8px; font-weight: 600; }}
  .card h2 span {{ color: #7f8fa1; font-weight: 400; }}
  .card img {{ width: 100%; border-radius: 6px; display: block; }}
  .counts {{ font-size: 12px; color: #9fb3c8; margin: 8px 0 4px; }}
  .missing {{ color: #d08c60; font-size: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 6px; }}
  th, td {{ text-align: left; padding: 2px 4px; border-bottom: 1px solid #2a323d; }}
  summary {{ cursor: pointer; font-size: 12px; color: #9fb3c8; }}
  tr.ok td {{ color: #8fd18f; }}
  tr.no td {{ color: #7f8fa1; }}
</style>
</head>
<body>
<h1>OpenCV 形状识别结果复查</h1>
<p class="summary">共 {len(payload)} 张图；全部检出物块：{html.escape(summary)}</p>
<div class="grid">{''.join(cards)}
</div>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(f"已写入 {output}（{len(payload)} 张图，物块合计 {sum(totals.values())} 个）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
