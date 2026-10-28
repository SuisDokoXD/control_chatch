"""读取复核页面导出的进度 JSON，算出你要的那个指标：检出且形状正确 / 真实块数。

用法：
    python scripts/ingest_review.py captures/shape_cv/review/review_progress.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="汇总复核结果")
    parser.add_argument("progress", type=Path, help="复核页面导出的 JSON")
    parser.add_argument("--json", type=Path, help="把逐图明细写到文件")
    args = parser.parse_args(argv)
    path = args.progress if args.progress.is_absolute() else ROOT / args.progress
    state = json.loads(path.read_text(encoding="utf-8"))

    total_true = total_ok = total_wrong = total_false = 0
    total_detected = 0
    rows = []
    for name in sorted(state):
        entry = state[name]
        count = entry.get("count")
        items = entry.get("items", {})
        ok = sum(1 for v in items.values() if v.get("verdict") == "ok")
        wrong = sum(1 for v in items.values() if v.get("verdict") == "wrong")
        false = sum(1 for v in items.values() if v.get("verdict") == "false")
        unlabeled = sum(1 for v in items.values() if not v.get("verdict"))
        detected = ok + wrong          # 认定是真物块的检出
        if count is None:
            rows.append((name, None, ok, wrong, false, unlabeled, None))
            continue
        total_true += count
        total_ok += ok
        total_wrong += wrong
        total_false += false
        total_detected += detected
        rows.append((name, count, ok, wrong, false, unlabeled,
                     ok / count if count else 0.0))

    print(f"{'图片':24}{'真实':>5}{'判对':>5}{'类别错':>7}{'误检':>5}{'未标':>5}{'检出且判对':>11}")
    for name, count, ok, wrong, false, unlabeled, rate in rows:
        if count is None:
            print(f"{name:24}{'未填':>5}{ok:>5}{wrong:>7}{false:>5}{unlabeled:>5}{'—':>11}")
            continue
        print(f"{name:24}{count:>5}{ok:>5}{wrong:>7}{false:>5}{unlabeled:>5}{rate * 100:>10.0f}%")

    print()
    if total_true:
        print(f"已复核 {sum(1 for r in rows if r[1] is not None)} 张图，真实物块合计 {total_true}")
        print(f"  检出且形状正确   {total_ok} / {total_true} = {total_ok / total_true:.1%}   ← 你要的指标")
        print(f"  检出但形状判错   {total_wrong} / {total_true} = {total_wrong / total_true:.1%}")
        print(f"  漏检            {max(total_true - total_detected, 0)} / {total_true} = "
              f"{max(total_true - total_detected, 0) / total_true:.1%}")
        print(f"  误检（不是物块） {total_false}")
        if total_detected:
            print(f"  算法给出的结论里判对的比例：{total_ok} / {total_ok + total_wrong} = "
                  f"{total_ok / (total_ok + total_wrong):.1%}")
    else:
        print("还没有填过'真实块数'，算不出比率。")
    if args.json:
        dest = args.json if args.json.is_absolute() else ROOT / args.json
        dest.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"明细已写入 {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
