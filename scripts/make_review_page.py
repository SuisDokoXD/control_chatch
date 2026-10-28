"""把 OpenCV 形状识别的结果导出成一个"看一眼就能复核"的页面。

用法：
    python scripts/make_review_page.py --limit 60
    python scripts/make_review_page.py --prefix awayfixed --limit 20

产出 captures/shape_cv/review/review.html：每张图显示原图 + 算法框（带编号和类别），
右侧只需要做两件事——填"真实块数"、点掉判错的框。支持导出/导入进度 JSON。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from 程序 import opencv_shape_vision as vision

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "captures" / "shape_cv" / "review"

TEMPLATE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>形状识别复核</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; background:#14171a; color:#e8eaed; font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif; }
  header { display:flex; align-items:center; gap:10px; padding:8px 14px; background:#1e2227;
           border-bottom:1px solid #2c3238; position:sticky; top:0; z-index:10; }
  header b { color:#8ab4f8; }
  button { background:#2c3238; color:#e8eaed; border:1px solid #3c444c; border-radius:6px;
           padding:5px 12px; cursor:pointer; font-size:13px; }
  button:hover { background:#3a4249; }
  button.primary { background:#1a73e8; border-color:#1a73e8; }
  button.primary:hover { background:#2b7de9; }
  main { display:flex; gap:14px; padding:14px; align-items:flex-start; }
  #stage { position:relative; flex:0 0 auto; }
  #stage img { display:block; width:min(62vw,1100px); border-radius:6px; }
  #overlay { position:absolute; inset:0; width:100%; height:100%; }
  .box { fill:none; stroke-width:3; }
  .box.ok { stroke:#34a853; }
  .box.wrong { stroke:#ea4335; }
  .box.unknown { stroke:#fbbc04; }
  .box.false { stroke:#9aa0a6; stroke-dasharray:8 6; }
  .tag { font:600 15px system-ui,sans-serif; paint-order:stroke; stroke:#000; stroke-width:4px; }
  .hit { fill:transparent; cursor:pointer; }
  aside { flex:1 1 auto; min-width:340px; max-width:520px; }
  .card { background:#1e2227; border:1px solid #2c3238; border-radius:8px; padding:12px; margin-bottom:10px; }
  .row { display:flex; align-items:center; gap:8px; padding:6px 0; border-bottom:1px solid #262c32; }
  .row:last-child { border-bottom:none; }
  .num { display:inline-flex; width:26px; height:26px; align-items:center; justify-content:center;
         border-radius:50%; background:#3c444c; font-size:13px; font-weight:600; flex:0 0 auto; }
  .algo { flex:0 0 auto; color:#fbbc04; font-family:ui-monospace,Consolas,monospace; font-size:12px; }
  .row button { padding:3px 9px; font-size:12px; }
  .row button.on-ok { background:#34a853; border-color:#34a853; }
  .row button.on-wrong { background:#ea4335; border-color:#ea4335; }
  .row button.on-false { background:#5f6368; border-color:#5f6368; }
  select { background:#2c3238; color:#e8eaed; border:1px solid #3c444c; border-radius:5px; padding:3px 6px; font-size:12px; }
  input[type=number] { background:#2c3238; color:#e8eaed; border:1px solid #3c444c; border-radius:5px;
  .box { fill:none; }
  .tag { font-weight:700; paint-order:stroke; stroke:#000; }
  .hit { fill:transparent; cursor:pointer; }
  .muted { color:#9aa0a6; font-size:12px; }
  .big { font-size:15px; }
  kbd { background:#2c3238; border:1px solid #3c444c; border-radius:4px; padding:1px 6px; font-size:12px; }
</style>
</head>
<body>
<header>
  <button id="prev">← 上一张</button>
  <button id="next">下一张 →</button>
  <b id="progress"></b>
  <span class="muted">跳到</span><input id="jump" type="number" min="1" style="width:70px">
  <span style="flex:1"></span>
  <button class="primary" id="export">导出进度 JSON</button>
  <button id="import">导入进度</button>
  <input type="file" id="file" accept=".json" hidden>
</header>
<main>
  <div id="stage">
    <img id="img" alt="">
    <svg id="overlay"></svg>
  </div>
  <aside>
    <div class="card">
      <div class="big">图片 <b id="name"></b></div>
      <div class="muted" id="filesize"></div>
      <div style="margin-top:10px">
        <span class="big">真实块数</span>
        <input id="count" type="number" min="0" placeholder="数一数">
        <span class="muted">（整盘里一共有几块）</span>
      </div>
      <div class="muted" style="margin-top:8px">
        快捷键：<kbd>1</kbd>=对 <kbd>2</kbd>=类别错 <kbd>3</kbd>=误检 ｜ <kbd>←</kbd><kbd>→</kbd> 翻页
      </div>
    </div>
    <div class="card" id="list"></div>
    <div class="card" id="summary"></div>
  </aside>
</main>
<script>
const DATA = __DATA__;
const CLASSES = __CLASSES__;
const KEY = "shape_review_progress_v1";
let state = JSON.parse(localStorage.getItem(KEY) || "{}");
let idx = 0;

const el = id => document.getElementById(id);
const names = Object.keys(DATA).sort();

function save() { localStorage.setItem(KEY, JSON.stringify(state)); }
function rec(name) {
  if (!state[name]) state[name] = { count: null, items: {} };
  return state[name];
}

function draw() {
  const name = names[idx];
  const entry = DATA[name];
  el("name").textContent = name;
  el("filesize").textContent = entry.width + "×" + entry.height + " ｜ 算法检出 " + entry.items.length + " 个";
  const img = el("img");
  img.src = entry.url;
  img.onload = () => {
    const svg = el("overlay");
    const w = img.clientWidth, h = img.clientHeight;
    svg.setAttribute("viewBox", `0 0 ${entry.width} ${entry.height}`);
    svg.setAttribute("preserveAspectRatio", "none");
    svg.style.width = w + "px"; svg.style.height = h + "px";
    svg.innerHTML = "";
    entry.items.forEach((it, i) => {
      const n = i + 1;
      const verdict = (rec(name).items[n] || {}).verdict || (it.label === "unknown" ? "unknown" : "");
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      r.setAttribute("x", it.bbox[0]); r.setAttribute("y", it.bbox[1]);
      r.setAttribute("width", it.bbox[2] - it.bbox[0]);
      r.setAttribute("height", it.bbox[3] - it.bbox[1]);
      const stroke = verdict === "wrong" ? "#ea4335" : verdict === "false" ? "#9aa0a6"
                   : verdict === "unknown" ? "#fbbc04" : "#34a853";
      const sw = entry.width / 420;
      r.setAttribute("stroke", stroke);
      r.setAttribute("stroke-width", sw);
      if (verdict === "false") r.setAttribute("stroke-dasharray", `${sw * 2.4} ${sw * 1.8}`);
      g.appendChild(r);
      const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("x", it.bbox[0] + entry.width / 300);
      t.setAttribute("y", Math.max(entry.width / 42, it.bbox[1] - entry.width / 150));
      t.setAttribute("class", "tag");
      t.setAttribute("font-size", entry.width / 42);
      t.setAttribute("stroke-width", entry.width / 260);
      t.setAttribute("fill", stroke);
      t.textContent = n + ":" + it.label;
      g.appendChild(t);
      const hit = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      hit.setAttribute("x", it.bbox[0]); hit.setAttribute("y", it.bbox[1]);
      hit.setAttribute("width", it.bbox[2] - it.bbox[0]);
      hit.setAttribute("height", it.bbox[3] - it.bbox[1]);
      hit.setAttribute("class", "hit");
      hit.addEventListener("click", () => setVerdict(n, "wrong"));
      g.appendChild(hit);
      svg.appendChild(g);
    });
  };
  el("count").value = rec(name).count ?? "";
  renderList();
  renderSummary();
  el("progress").textContent = `${idx + 1} / ${names.length}`;
  el("jump").value = idx + 1;
}

function renderList() {
  const name = names[idx];
  const entry = DATA[name];
  const box = el("list");
  box.innerHTML = "";
  if (!entry.items.length) {
    box.innerHTML = '<div class="muted">算法在这一张里没有任何检出 —— 请把真实块数填上。</div>';
    return;
  }
  entry.items.forEach((it, i) => {
    const n = i + 1;
    const cur = rec(name).items[n] || {};
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<span class="num">${n}</span><span class="algo">${it.label}</span>`;
    const mk = (text, val, cls) => {
      const b = document.createElement("button");
      b.textContent = text;
      if (cur.verdict === val) b.className = cls;
      b.onclick = () => setVerdict(n, cur.verdict === val ? "" : val);
      return b;
    };
    row.appendChild(mk("对", "ok", "on-ok"));
    row.appendChild(mk("类别错", "wrong", "on-wrong"));
    row.appendChild(mk("误检", "false", "on-false"));
    if (cur.verdict === "wrong") {
      const sel = document.createElement("select");
      sel.innerHTML = '<option value="">正确类别…</option>' +
        CLASSES.map(c => `<option ${cur.true === c ? "selected" : ""}>${c}</option>`).join("");
      sel.onchange = () => { const r = rec(name).items[n]; r.true = sel.value; save(); };
      row.appendChild(sel);
    }
    box.appendChild(row);
  });
}

function renderSummary() {
  const name = names[idx];
  const e = rec(name);
  const items = Object.values(e.items);
  const ok = items.filter(v => v.verdict === "ok").length;
  const wrong = items.filter(v => v.verdict === "wrong").length;
  const fals = items.filter(v => v.verdict === "false").length;
  const n = DATA[name].items.length;
  const cnt = e.count;
  let line = `已标：对 ${ok} ｜ 类别错 ${wrong} ｜ 误检 ${fals} ｜ 未标 ${n - ok - wrong - fals}`;
  if (cnt !== null && cnt !== "") {
    const miss = cnt - ok - wrong;
    line += `<br>真实块数 ${cnt} → <b>检出且判对 ${ok} / ${cnt} = ${(ok / Math.max(cnt, 1) * 100).toFixed(0)}%</b>`;
    line += `<br>漏检 ${miss > 0 ? miss : 0}`;
  }
  el("summary").innerHTML = line;
}

function setVerdict(n, v) {
  const name = names[idx];
  const r = rec(name);
  r.items[n] = r.items[n] || {};
  r.items[n].verdict = v;
  if (v !== "wrong") delete r.items[n].true;
  save(); renderList(); renderSummary(); draw();
}

el("count").addEventListener("input", e => {
  rec(names[idx]).count = e.target.value === "" ? null : Number(e.target.value);
  save(); renderSummary();
});
el("prev").onclick = () => { if (idx > 0) { idx--; draw(); } };
el("next").onclick = () => { if (idx < names.length - 1) { idx++; draw(); } };
el("jump").onchange = e => {
  const v = Number(e.target.value);
  if (v >= 1 && v <= names.length) { idx = v - 1; draw(); }
};
el("export").onclick = () => {
  const blob = new Blob([JSON.stringify(state, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "review_progress.json";
  a.click();
};
el("import").onclick = () => el("file").click();
el("file").onchange = e => {
  const f = e.target.files[0];
  if (!f) return;
  const fr = new FileReader();
  fr.onload = () => { state = JSON.parse(fr.result); save(); draw(); };
  fr.readAsText(f);
};
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "ArrowLeft") el("prev").click();
  if (e.key === "ArrowRight") el("next").click();
});
draw();
</script>
</body>
</html>
"""


def build(limit: int, prefix: str | None, output: Path) -> Path:
    images = sorted(vision.RAW_IMAGES.glob("*.jpg"))
    if prefix:
        images = [p for p in images if p.stem.startswith(prefix)]
    if limit > 0:
        step = max(1, len(images) // limit)
        images = images[::step][:limit]

    data: dict[str, dict] = {}
    for path in images:
        image = vision.cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        _, objects = vision.analyse_image(image)
        items = []
        for obj in objects:
            x1, y1, x2, y2 = (int(v / vision.DEFAULTS["scale"]) for v in obj.bbox)
            items.append({"label": obj.label, "bbox": [x1, y1, x2, y2]})
        rel = Path("..", "..", "..", "dataset", "yolo_cube", "images", "raw", path.name)
        data[path.name] = {"url": rel.as_posix(), "width": width, "height": height, "items": items}

    output.mkdir(parents=True, exist_ok=True)
    html = (TEMPLATE
            .replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__CLASSES__", json.dumps(list(vision.SHAPES), ensure_ascii=False)))
    dest = output / "review.html"
    dest.write_text(html, encoding="utf-8")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成形状识别复核页面")
    parser.add_argument("--limit", type=int, default=60, help="抽多少张（按等间隔抽样）")
    parser.add_argument("--prefix", type=str, default=None, help="只看某个前缀，如 awayfixed")
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    dest = build(args.limit, args.prefix, output)
    print(f"已生成 {dest}")
    print("用浏览器打开它：填'真实块数'、点掉判错的框，然后导出进度 JSON。")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
