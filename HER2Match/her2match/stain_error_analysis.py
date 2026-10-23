#!/usr/bin/env python3
"""Reproducible image-level error proxies. Never assigns clinical HER2 scores.

DAB is skimage's fixed HED color deconvolution (natural-log optical density).
Thresholds are reference quantiles on native-resolution REAL VALIDATION images
only, equally sampled by available validation slide. They are descriptive image
thresholds, not a stain-normalized assay, membrane detector, tumor-cell counter,
or pathology-grade segmentation. No geometric registration is performed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import sobel, binary_opening, binary_closing
from skimage.color import rgb2hed, rgb2hsv, rgb2gray


def read_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'))


def _float_rgb(rgb):
    x = np.asarray(rgb)
    x = x.astype(np.float32) / (255.0 if np.issubdtype(x.dtype, np.integer) else 1.0)
    if x.ndim != 3 or x.shape[2] != 3 or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError('Expected finite HxWx3 RGB uint8 or float in [0,1]')
    return x


def tissue_mask(rgb):
    """Permissive optical tissue mask; includes colored or dark nonwhite pixels.

    Fixed thresholds, not fitted on test images. Small 3x3 opening/closing removes
    isolated pixels. This is NOT tumor/cell segmentation; inspect mask coverage.
    """
    x = _float_rgb(rgb)
    hsv = rgb2hsv(x)
    mask = ((hsv[..., 1] > .08) & (x.mean(-1) < .97)) | (x.mean(-1) < .90)
    return binary_closing(binary_opening(mask, iterations=1), iterations=1)


def _channels(rgb):
    x = _float_rgb(rgb)
    hed = rgb2hed(x).astype(np.float32)
    return x, np.maximum(hed[..., 2], 0), np.maximum(hed[..., 0], 0)


def _mean(x, mask):
    return float(np.mean(x[mask])) if np.any(mask) else None


def _fraction(mask, region):
    return _mean(mask.astype(np.float32), region)


def calibrate(records, data_root, max_per_group=12):
    """Calibrate on evenly spaced validation records within each slide.

    At most 8192 deterministic tissue pixels/image; groups and images receive
    equal pixel budgets when enough tissue is present. Empty tissue is skipped
    and documented. The record list can be the full original manifest.
    """
    root = Path(data_root)
    groups = {}
    for record in records:
        if record.get('split') in ('val', 'validation'):
            group = record.get('group')
            if group is None:
                raise ValueError('Validation record missing slide/group identity')
            groups.setdefault(str(group), []).append(record)
    if not groups:
        raise ValueError('No real validation records; refusing test calibration')
    values, selected, skipped = [], [], []
    group_counts = {}
    for group, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row['id'])
        indices = np.linspace(0, len(rows)-1, min(max_per_group, len(rows)), dtype=int)
        for idx in indices:
            row = rows[idx]
            rgb = read_rgb(root / row['ihc'])
            _, dab, _ = _channels(rgb)
            pixels = dab[tissue_mask(rgb)]
            if pixels.size < 256:
                skipped.append(row['id'])
                continue
            # Equal pixel count per included image; repeat sparse tissue deterministically.
            pixels = pixels[np.linspace(0, pixels.size-1, 8192, dtype=int)]
            values.append(pixels)
            selected.append(row['id'])
            group_counts[group] = group_counts.get(group, 0) + 1
    if not values:
        raise ValueError('No usable tissue in validation reference images')
    pool = np.concatenate(values)
    q = np.quantile(pool, [.25, .5, .75, .9, .95, .99])
    identity = '\n'.join(selected).encode()
    return {
        'schema_version': 1, 'fit_split': 'validation_only',
        'source': 'real HER2 ground truth; fixed skimage HED deconvolution',
        'sample_ids': selected, 'sample_ids_sha256': hashlib.sha256(identity).hexdigest(),
        'groups': group_counts, 'skipped_low_tissue_ids': skipped,
        'sampling': 'evenly spaced IDs per slide; 8192 equally spaced tissue pixels per image',
        'dab_quantiles': dict(zip(['q25','q50','q75','q90','q95','q99'], map(float, q))),
        'threshold_rule': 'DAB=max(.02,val_tissue_q75); strong=max(.04,val_tissue_q90,DAB+.01). Floors fixed before test analysis to suppress nearzero deconvolution noise.',
        'dab_threshold': max(.02, float(q[2])),
        'dab_strong_threshold': max(.04, float(q[3]), max(.02, float(q[2])) + .01),
        'tissue_mask': '((HSV saturation > .08) AND (mean RGB < .97)) OR mean RGB < .90; 3x3 opening then closing',
        'limitations': 'Image stain proxies only, not clinical scores or positive tumor-cell fractions. Fixed HED axes may not separate stains perfectly. No registration. Spatial errors are alignment-sensitive. Reference thresholds may transfer poorly across datasets.',
    }


def analyze_pair(gt_rgb, pred_rgb, calibration, he_rgb=None):
    """Return JSON-safe scalar proxies; native resolution, no resize/registration."""
    if np.shape(gt_rgb) != np.shape(pred_rgb):
        raise ValueError('Prediction and ground truth dimensions differ; no implicit resizing')
    gt, gd, gh = _channels(gt_rgb)
    pred, pd, ph = _channels(pred_rgb)
    mask, pmask = tissue_mask(gt_rgb), tissue_mask(pred_rgb)
    bg = ~mask
    union = mask | pmask
    threshold = calibration['dab_threshold']
    strong = calibration['dab_strong_threshold']
    gp, pp = gd > threshold, pd > threshold
    n = int(mask.sum())
    inter = (gp & pp & mask).sum()
    positive_union = ((gp | pp) & mask).sum()
    ggrad = np.hypot(sobel(rgb2gray(gt), axis=0), sobel(rgb2gray(gt), axis=1))
    pgrad = np.hypot(sobel(rgb2gray(pred), axis=0), sobel(rgb2gray(pred), axis=1))
    result = {
        'height': int(gt.shape[0]), 'width': int(gt.shape[1]), 'gt_tissue_pixels': n,
        'gt_tissue_fraction': float(mask.mean()), 'pred_tissue_fraction': float(pmask.mean()),
        'tissue_mask_iou': float((mask & pmask).sum()/union.sum()) if union.any() else None,
        'gt_dab_mean': _mean(gd, mask), 'pred_dab_mean': _mean(pd, mask),
        'dab_mean_bias': _mean(pd-gd, mask), 'dab_mae': _mean(np.abs(pd-gd), mask),
        'gt_dab_coverage': _fraction(gp, mask), 'pred_dab_coverage': _fraction(pp, mask),
        'gt_strong_dab_coverage': _fraction(gd > strong, mask),
        'pred_strong_dab_coverage': _fraction(pd > strong, mask),
        'understain_area_fraction': _fraction(gp & ~pp, mask),
        'overstain_area_fraction': _fraction(~gp & pp, mask),
        'dab_mask_iou': float(inter/positive_union) if positive_union else None,
        'hematoxylin_mean_bias': _mean(ph-gh, mask),
        'background_pred_tissue_fraction': _fraction(pmask, bg),
        'background_dab_mean': _mean(pd, bg),
        'background_rgb_mae': _mean(np.abs(pred-gt).mean(-1), bg),
        'gt_edge_strength': _mean(ggrad, mask), 'pred_edge_strength': _mean(pgrad, mask),
        'edge_strength_ratio': None,
        'edge_map_mae': _mean(np.abs(pgrad-ggrad), mask),
        'pred_tissue_channel_black_fraction': _mean((pred <= 0).mean(-1), mask),
        'pred_tissue_channel_white_fraction': _mean((pred >= 1).mean(-1), mask),
    }
    if result['gt_edge_strength'] is not None and result['gt_edge_strength'] > 0:
        result['edge_strength_ratio'] = result['pred_edge_strength']/result['gt_edge_strength']
    for j, channel in enumerate('rgb'):
        result[f'{channel}_mean_bias'] = _mean(pred[...,j]-gt[...,j], mask)
    if he_rgb is not None:
        he = _float_rgb(he_rgb)
        if he.shape != gt.shape:
            raise ValueError('H&E dimensions differ')
        result['pred_he_rgb_mae'] = _mean(np.abs(pred-he).mean(-1), mask)
        result['gt_he_rgb_mae'] = _mean(np.abs(gt-he).mean(-1), mask)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['calibrate','analyze'])
    p.add_argument('--manifest', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--calibration')
    p.add_argument('--predictions')
    p.add_argument('--split', default='test')
    p.add_argument('--max-per-group', type=int, default=12)
    args = p.parse_args()
    records = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.action == 'calibrate':
        out.write_text(json.dumps(calibrate(records,args.data_root,args.max_per_group),indent=2)+'\n')
        return
    if not args.calibration or not args.predictions:
        p.error('analyze requires --calibration and --predictions')
    cal = json.loads(Path(args.calibration).read_text())
    root, predictions = Path(args.data_root), Path(args.predictions)
    rows = [r for r in records if r.get('split') == args.split]
    if not rows:
        raise ValueError('No records for requested split')
    # Preflight prevents silently reporting metrics for an incomplete selection.
    missing = [r['id'] for r in rows if not (predictions/(r['id']+'.png')).exists()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} predictions missing; first: {missing[0]}')
    with out.open('w',newline='') as f:
        writer = None
        for idx,row in enumerate(rows):
            metrics = analyze_pair(read_rgb(root/row['ihc']),read_rgb(predictions/(row['id']+'.png')),cal,read_rgb(root/row['he']))
            record = dict(id=row['id'],group=row.get('group'),her2_score=row.get('her2_score'),**metrics)
            if writer is None:
                writer = csv.DictWriter(f,fieldnames=list(record));writer.writeheader()
            writer.writerow(record)
            if (idx+1)%100 == 0:
                f.flush();print(f'{idx+1}/{len(rows)} analyzed',flush=True)

if __name__ == '__main__':
    main()
