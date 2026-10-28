"""Summarise recorded real-camera evidence, explicitly leaving human accuracy unknown."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.two_view_live import atomic_json


def read_lines(path):
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # Only a last line being written by the live process may be incomplete.
            if line != path.read_text(encoding='utf-8').splitlines()[-1]:
                raise
    return rows


def percentiles(values):
    return dict(zip(('p50','p95','max'), map(float, np.percentile(values,[50,95,100])))) if values else None


def summarise(session):
    telemetry = read_lines(session / 'telemetry.jsonl')
    fits = read_lines(session / 'fits.jsonl')
    targets, labels, missing = [], [], 0
    for fit in fits:
        nearby = []
        for item in fit['detections']:
            point = item['position_mm'] or item.get('unverified_estimates',{}).get('position_mm')
            if point and np.hypot(point['x']-90,point['y']-60)<15:
                nearby.append(item)
        if not nearby:
            missing += 1
            labels.append('missing')
            continue
        target = nearby[0]
        labels.append(target['class'])
        if target['geometry_valid']:
            targets.append([target['position_mm']['x'],target['position_mm']['y'],target['dimensions_mm']['height']])
    coordinates = {}
    if targets:
        data = np.asarray(targets)
        coordinates = {axis:{'mean_mm':float(data[:,i].mean()),'std_mm':float(data[:,i].std()),
                             'peak_to_peak_mm':float(np.ptp(data[:,i]))}
                       for i,axis in enumerate(('x','y','height'))}
    classes = Counter(d['class'] for fit in fits for d in fit['detections'])
    count = sum(classes.values())
    visible_target = sum(any(d['class']=='pentagonal_prism' for d in r['detections']) for r in telemetry)
    states = Counter(row['state'] for row in telemetry)
    session_data = json.loads((session/'session.json').read_text(encoding='utf-8'))
    metrics = {
        'session':str(session.resolve()), 'cameras':session_data['cameras'],
        'calibration':session_data['calibration'], 'chinese_font_available':session_data['chinese_font'],
        'telemetry_records':len(telemetry),'telemetry_span_s':telemetry[-1]['elapsed_s']-telemetry[0]['elapsed_s'],
        'fit_count':len(fits),'fit_source_span_s':fits[-1]['source_s']-fits[0]['source_s'],
        'states':dict(states),'fit_ms':percentiles([r['fit_ms'] for r in fits]),
        'loop_ms':percentiles([r['loop_ms'] for r in telemetry]),
        'frame_gap_ms':percentiles([r['frame_gap_ms'] for r in telemetry]),
        'segmentation_count_frequency':dict(Counter(r['segmentation_count'] for r in telemetry)),
        'fit_class_frequency':dict(classes),'class_rejection_rate':classes['unknown(unclassified)']/count if count else None,
        'tracked_seed_mm':[90,60], 'tracked_fit_labels':dict(Counter(labels)),
        'tracked_class_switches':sum(a!=b for a,b in zip(labels,labels[1:])),
        'tracked_fit_missing':missing,'tracked_fit_presence_rate':(len(fits)-missing)/len(fits),
        'tracked_visible_fraction':visible_target/len(telemetry),
        'tracked_coordinates':coordinates,'saved_sample_pairs':len(list((session/'samples').iterdir())),
        'human_ground_truth_available':False,'absolute_dimension_accuracy_verified':False,
        'classification_accuracy':None,'detection_recall':None,'extra_detection_count':None,
        'limitation':'Tracking seed is a diagnostic location, not a human-confirmed current object label; no robot execution.',
    }
    atomic_json(session/'metrics.json',metrics)
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--session',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(summarise(args.session),ensure_ascii=False,indent=2))