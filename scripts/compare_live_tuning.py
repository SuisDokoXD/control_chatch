"""Reproduce original/revised results on identical saved real frames; no fabricated GT."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from scripts import two_view_shape as revised
from scripts.two_view_live import atomic_json, segment_pair


def load_original(path):
    spec = importlib.util.spec_from_file_location('original_two_view_shape', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(root, session, limit):
    original = load_original(root / 'two_view_shape.py')
    cal = json.loads((ROOT / '配置/calibration_stereo.json').read_text(encoding='utf-8'))
    refs = {role: cv2.imread(str(ROOT / f'captures/calibration/ref_{role}.jpg'))
            for role in ('vertical', 'side_a')}
    fixture = cv2.imread(str(ROOT / 'captures/calibration/ref_static_vertical.png'), 0)
    if fixture is not None:
        fixture = cv2.dilate((fixture > 0).astype(np.uint8), np.ones((15,15), np.uint8))
    output = root / 'comparison'
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for folder in sorted((session / 'samples').iterdir())[:limit]:
        frames = {role: cv2.imread(str(folder / f'{role}.jpg')) for role in refs}
        row = {'sample': folder.name, 'human_ground_truth': None}
        for name, api in (('before', original), ('after', revised)):
            start = time.perf_counter()
            masks = segment_pair(api, frames, refs, cal, 28, 2500, fixture)
            segmentation_ms = (time.perf_counter()-start)*1000
            detections = api.analyse_masks(masks['vertical'], masks['side_a'], cal,
                                          min_area=2500, image_vertical=frames['vertical'])
            payload = api.detections_to_payload(detections)
            elapsed = (time.perf_counter()-start)*1000
            target = output / folder.name / name
            target.mkdir(parents=True, exist_ok=True)
            for role, mask in masks.items():
                cv2.imwrite(str(target / f'{role}_mask.png'), mask * 255)
            annotated = frames['vertical'].copy()
            for detection in detections:
                cv2.drawContours(annotated, [detection.contour_px], -1, (0,220,0) if detection.geometry_verified else (0,150,255), 3)
                x,y,_,_ = cv2.boundingRect(detection.contour_px)
                label = detection.label if detection.label in revised.SHAPE_CLASSES else 'tentative:' + detection.shape_hypothesis
                cv2.putText(annotated, label, (max(0,x),max(30,y-10)), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,255), 2)
            cv2.imwrite(str(target / 'overlay.jpg'), cv2.resize(annotated,(1296,972)))
            component_counts = {}
            for role, mask in masks.items():
                _,_,stats,_ = cv2.connectedComponentsWithStats(mask)
                component_counts[role] = int(np.count_nonzero(stats[1:,4] >= 2500))
            row[name] = {'component_counts': component_counts, 'segmentation_ms': segmentation_ms,
                         'total_ms': elapsed, 'detections': payload,
                         'confirmed_count':sum(d.label in revised.SHAPE_CLASSES for d in detections),
                         'rejected_count':sum(d.label not in revised.SHAPE_CLASSES for d in detections),
                         'valid_geometry_count':sum(d.geometry_verified for d in detections)}
            atomic_json(target / 'result.json', row[name])
        # Compare model-confirmed targets, not unknowable human correctness.
        prior = [d for d in row['before']['detections'] if d['class'] in revised.SHAPE_CLASSES]
        after = row['after']['detections']
        retained = 0
        for d in prior:
            p = d['position_mm'] or d.get('unverified_estimates',{}).get('position_mm')
            if any(a['class']==d['class'] and a['position_mm'] is not None
                   and np.hypot(a['position_mm']['x']-p['x'],a['position_mm']['y']-p['y'])<8 for a in after):
                retained += 1
        row['previously_confirmed_retained'] = retained
        row['previously_confirmed_count'] = len(prior)
        records.append(row)
        atomic_json(output / 'results.json', records)
        print(json.dumps({'sample':folder.name,
                          'before_confirmed':row['before']['confirmed_count'],
                          'after_confirmed':row['after']['confirmed_count'],
                          'before_components':row['before']['component_counts'],
                          'after_components':row['after']['component_counts']},ensure_ascii=False),flush=True)
    summary = {'samples':len(records), 'human_ground_truth_available':False,
               'detection_recall':None,'classification_accuracy':None,'extra_detections':None,
               'reason':'Image pixels unavailable to text-only model; no current human-verified object list.',
               'before_confirmed':sum(r['before']['confirmed_count'] for r in records),
               'after_confirmed':sum(r['after']['confirmed_count'] for r in records),
               'before_rejected':sum(r['before']['rejected_count'] for r in records),
               'after_rejected':sum(r['after']['rejected_count'] for r in records),
               'previously_confirmed':sum(r['previously_confirmed_count'] for r in records),
               'previously_confirmed_retained':sum(r['previously_confirmed_retained'] for r in records)}
    atomic_json(output/'summary.json',summary)
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT/'reports/live_tuning_20261027')
    p.add_argument('--session',type=Path,default=ROOT/'reports/live_tuning_20261027/run01')
    p.add_argument('--limit',type=int,default=40)
    args=p.parse_args()
    print(json.dumps(run(args.root,args.session,args.limit),ensure_ascii=False))