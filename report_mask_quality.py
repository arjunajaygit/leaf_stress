"""Read-only dataset inspection. Reports each byte-unique image and all gate reasons."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import config
from feature_extraction import prepare_leaf, QUALITY_MESSAGES


def inspect_dataset(dataset_dir):
    groups = defaultdict(list)
    for path in sorted(Path(dataset_dir).glob('*/*')):
        if path.suffix.lower() in config.ALLOWED_EXTENSIONS:
            groups[hashlib.sha256(path.read_bytes()).hexdigest()].append(path)
    rows = []
    for digest, paths in groups.items():
        path = paths[0]
        *_, quality = prepare_leaf(path)
        rows.append({
            'filename': str(path.relative_to(dataset_dir)), 'label': path.parent.name,
            'sha256': digest,
            'duplicate_filenames': ';'.join(str(p.relative_to(dataset_dir)) for p in paths[1:]),
            'border_contact': quality.get('border_contact', ''),
            'foreground_fraction': quality.get('foreground_fraction', ''),
            'candidate_foreground_fraction': quality.get('candidate_foreground_fraction', ''),
            'components': quality.get('components', ''), 'large_components': quality.get('large_components', ''),
            'contour_area': quality.get('contour_area', ''),
            'working_width': quality.get('working_width', ''), 'working_height': quality.get('working_height', ''),
            'pass_fail': 'pass' if quality['passed'] else 'fail',
            'reason_code': quality['reason_code'], 'reason_codes': ';'.join(quality['reason_codes']),
            'message': quality['message'],
        })
    return rows


def write_reports(rows, csv_path, summary_path):
    csv_path, summary_path = Path(csv_path), Path(summary_path)
    if not rows:
        raise ValueError('No dataset images found')
    if csv_path.exists() or summary_path.exists():
        raise FileExistsError('Reports already exist; choose new versioned output paths to preserve previous results.')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    reasons = Counter(r for row in rows for r in row['reason_codes'].split(';') if r)
    primary = Counter(row['reason_code'] for row in rows if row['pass_fail'] == 'fail')
    passed = sum(row['pass_fail'] == 'pass' for row in rows)
    lines = ['# Mask quality review — Stage 3', '',
             f'{len(rows)} byte-unique images: **{passed} pass, {len(rows)-passed} fail**. No images were removed.', '',
             'Provisional thresholds are acquisition checks, not a validated leaf detector. A pass does not establish leaf identity, label correctness or water status.', '',
             'Counts below include every reason per image, so they overlap and need not sum to the rejected-image count. Primary counts use the deterministic first reason only.', '',
             '| Reason | Images failing this reason | Primary reason count |', '|---|---:|---:|']
    lines += [f'| {reason} | {reasons[reason]} | {primary[reason]} |' for reason in QUALITY_MESSAGES]
    lines += ['', '| Class | Pass | Fail |', '|---|---:|---:|']
    for label in sorted({r['label'] for r in rows}):
        class_rows = [r for r in rows if r['label'] == label]
        yes = sum(r['pass_fail'] == 'pass' for r in class_rows)
        lines.append(f'| {label} | {yes} | {len(class_rows)-yes} |')
    lines += ['', '## Threshold snapshot', '', '```json',
              json.dumps({'preprocessing': config.preprocessing_signature(), 'quality': config.mask_quality_settings()}, indent=2), '```', '',
              'Foreground fraction and border contact refer to the selected filled mask. Components are measured on the full post-morphology candidate mask before choosing the largest contour. A component is large only if its pixel area reaches both the configured image-area fraction and the configured fraction of the largest component.', '',
              'All images are resized to longest side 800, including upsampling smaller images. The normal morphology kernel is 5×5 at that resolution; smaller direct segmentation inputs receive a scaled odd kernel. Upsampling adds no detail.', '',
              '## Rejected images', '', '| Filename | All reason codes |', '|---|---|']
    lines += [f"| {r['filename']} | {r['reason_codes']} |" for r in rows if r['pass_fail'] == 'fail']
    lines += ['', 'Byte-identical aliases are listed in the CSV. Distinct photography sessions/plants are unknown. Border rejection may be strict for isolated leaves whose stems touch the frame; review the images rather than changing thresholds solely to increase acceptance.', '']
    summary_path.write_text('\n'.join(lines))
    return {'pass': passed, 'fail': len(rows)-passed, 'reason_counts': dict(reasons)}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=config.BASE_DIR / 'reports/mask_quality_report.csv')
    parser.add_argument('--summary', type=Path, default=config.BASE_DIR / 'reports/mask_quality_summary.md')
    args = parser.parse_args()
    print(json.dumps(write_reports(inspect_dataset(config.DATASET_DIR), args.csv, args.summary), indent=2))
