"""Grouped nested-CV feature experiments. Never writes or selects a production model."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo
import cv2
import numpy as np
import pandas as pd
import config
from build_dataset import prepare_dataset
from dataset_manifest import FEATURE_NAMES, CLASSES
from model_contract import library_versions
from evaluate_grouped import readiness, nested_plan, evaluate_nested, candidate_models, BLOCKED_MESSAGE
from feature_extraction import (prepare_leaf, extract_color_features, extract_shape_features,
                                extract_experimental_features)

VARIANTS = {
    'baseline_15': list(FEATURE_NAMES),
    'without_exg': [f for f in FEATURE_NAMES if f != 'exg_mean'],
    'plus_chromaticity': [*FEATURE_NAMES, 'chrom_r_mean', 'chrom_g_mean'],
    'plus_lab': [*FEATURE_NAMES, 'lab_a_mean', 'lab_a_std', 'lab_b_mean', 'lab_b_std'],
    'oriented_ratio': [f if f != 'aspect_ratio' else 'oriented_axis_ratio' for f in FEATURE_NAMES],
}


def synthetic_frame():
    """Known invented groups/labels, solely to exercise code, not biological evidence."""
    rng = np.random.default_rng(42)
    rows = []
    for group in range(6):
        for label_index, label in enumerate(CLASSES):
            image = np.full((100, 100, 3), 240, np.uint8)
            mask = np.zeros((100, 100), np.uint8)
            cv2.ellipse(mask, (50, 50), (20 + label_index, 32), group * 12, 0, 360, 255, -1)
            color = np.array([35 + label_index * 12, 170 - label_index * 15, 45 + label_index * 18])
            image[mask == 255] = np.clip(color + rng.normal(0, 8, (np.count_nonzero(mask), 3)), 0, 255).astype(np.uint8)
            contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
            rows.append({**extract_color_features(image, mask), **extract_shape_features(contour),
                         **extract_experimental_features(image, mask, contour),
                         'label': label, 'filename': f'SYNTHETIC-{group}-{label}',
                         'group_id': f'SYNTHETIC-plant-{group}', 'sha256': f'SYNTHETIC-{group}-{label}'})
    return pd.DataFrame(rows)


def run_ablation(frame, candidates=None):
    nested_plan(frame)  # Preflight before fitting any variant.
    return {name: evaluate_nested(frame, candidates if candidates is not None else candidate_models(), columns)
            for name, columns in VARIANTS.items()}


def real_frame(manifest):
    frame, audit = prepare_dataset(manifest, config.DATASET_DIR)
    deficits = readiness(frame)
    if any(row['missing'] for row in deficits.values()):
        return frame, audit, deficits
    for index, row in frame.iterrows():
        image, mask, contour, _, quality = prepare_leaf(config.DATASET_DIR / row.filename)
        if not quality['passed']:
            raise ValueError('Gate changed during ablation preparation')
        for name, value in extract_experimental_features(image, mask, contour).items():
            frame.loc[index, name] = value
    return frame, audit, deficits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--manifest', type=Path, default=config.BASE_DIR / 'dataset_manifest.csv')
    parser.add_argument('--output-dir', type=Path, default=config.BASE_DIR / 'reports')
    args = parser.parse_args(argv)
    payload = {'library_versions': library_versions(), 'protocol': {'outer_folds': 3, 'inner_folds': 3, 'seed': 42, 'grouped': True}, 'synthetic': args.synthetic, 'production_features_changed': False,
               'variants': VARIANTS, 'results': None,
               'interpretation': 'Each variant tunes models only inside inner grouped folds and uses identical outer partitions. Comparing variants on outer results is exploratory, not an unbiased estimate after selecting a feature set. No feature or production-model selection is performed.'}
    if args.synthetic:
        frame = synthetic_frame()
        payload['warning'] = 'SYNTHETIC SOFTWARE VALIDATION ONLY — invented plants, labels and pixels. Not project performance.'
    else:
        frame, payload['audit'], payload['per_class_readiness'] = real_frame(args.manifest)
    try:
        nested_plan(frame)
    except ValueError as exc:
        payload.update(status='blocked', message=BLOCKED_MESSAGE, detail=str(exc))
    else:
        payload['results'] = run_ablation(frame)
        payload['status'] = 'completed'
    when = datetime.now(ZoneInfo('Asia/Kolkata'))
    payload['generated_at'] = when.isoformat()
    prefix = 'ablation_synthetic' if args.synthetic else 'ablation'
    stem = f'{prefix}_{when:%Y-%m-%d_%H%M%S}_{uuid4().hex[:8]}'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / (stem + '.json')).open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    lines = ['# Feature ablations', '', payload.get('warning', payload.get('message', 'Grouped exploratory ablations')), '', payload['interpretation'], '']
    if payload['results']:
        lines += ['| Variant | Outer Macro F1 | Balanced accuracy |', '|---|---:|---:|']
        for name, result in payload['results'].items():
            m = result['selected_procedure_metrics']
            lines.append(f"| {name} | {m['macro_f1']} | {m['balanced_accuracy']} |")
    else:
        lines += ['No evaluation numbers produced. No models saved.', '', json.dumps(payload.get('per_class_readiness', {}), indent=2)]
    lines += ['', 'Feature definitions: additions are independent comparisons against the baseline, not cumulative. Chromaticity is mean pixelwise R/(R+G+B), G/(R+G+B), with black pixels assigned zero. LAB uses floating-point OpenCV CIELAB a*/b* means and population standard deviations. Oriented ratio is minAreaRect short/long side, replacing the axis-aligned ratio. No claim of illumination invariance or direct curl measurement.', '', 'Full metrics and fold memberships are in the companion JSON. Production features and model remain unchanged.']
    path = args.output_dir / (stem + '.md')
    with path.open('x') as stream: stream.write('\n'.join(lines))
    print(payload.get('warning', payload.get('message', payload['status'])))
    print(path)
    return 2 if payload['status'] == 'blocked' else 0


if __name__ == '__main__':
    raise SystemExit(main())
