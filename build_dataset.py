"""Build gate-passing, byte-deduplicated features without import-time side effects."""
import argparse
import json
from pathlib import Path

import pandas as pd
import config
from dataset_manifest import CLASSES, FEATURE_NAMES, deduplicate, load_manifest, resolve_groups, sha256
from feature_extraction import prepare_leaf

MIN_BUILD_SAMPLES_PER_CLASS = 2  # Feasibility guard only, not a sufficiency claim.


def prepare_dataset(manifest_path, dataset_dir):
    original = load_manifest(manifest_path, dataset_dir)
    unique = deduplicate(original)  # Conflicts are checked even in rejected images.
    # Resolve identities using all records so a rejected bridge cannot hide shared identity.
    identities = resolve_groups(unique)
    rows, rejected = [], []
    for row, (group, source) in zip(unique, identities):
        *_, features, quality = prepare_leaf(Path(dataset_dir) / row['filename'])
        if not quality['passed']:
            rejected.append({'filename': row['filename'], 'label': row['class_label'], 'reason_codes': quality['reason_codes']})
            continue
        rows.append({**features, 'label': row['class_label'], 'filename': row['filename'],
                     'sha256': row['sha256'], 'plant_id': row['plant_id'], 'leaf_id': row['leaf_id'],
                     'session_id': row['session_id'], 'group_id': group, 'group_source': source})
    columns = [*FEATURE_NAMES, 'label', 'filename', 'sha256', 'plant_id', 'leaf_id', 'session_id', 'group_id', 'group_source']
    frame = pd.DataFrame(rows, columns=columns)
    return frame, {'raw_image_files': len(original), 'unique_images': len(unique),
                   'duplicates_removed': len(original) - len(unique), 'rejected': rejected,
                   'manifest_sha256': sha256(manifest_path)}


def class_counts(frame):
    return {label: int((frame['label'] == label).sum()) for label in CLASSES}


def build_dataset(manifest_path, dataset_dir, output):
    frame, audit = prepare_dataset(manifest_path, dataset_dir)
    counts = class_counts(frame)
    missing = {label: {'accepted': n, 'minimum': MIN_BUILD_SAMPLES_PER_CLASS,
                       'additional_needed': max(0, MIN_BUILD_SAMPLES_PER_CLASS - n)}
               for label, n in counts.items() if n < MIN_BUILD_SAMPLES_PER_CLASS}
    if missing:
        raise ValueError('Too few gate-passing samples; no feature artifact written: ' + json.dumps(missing))
    output = Path(output)
    if output.exists() or output.with_suffix('.metadata.json').exists():
        raise FileExistsError('Choose a new versioned artifact filename; existing outputs are preserved')
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, mode='x')
    with output.with_suffix('.metadata.json').open('x') as stream:
        json.dump({'preprocessing': config.preprocessing_signature(), 'mask_quality': config.mask_quality_settings(),
                   'csv_sha256': sha256(output), 'audit': audit,
                   'feature_names': FEATURE_NAMES, 'groups_available': bool(frame['group_id'].ne('').all())}, stream, indent=2)
    if frame['group_id'].eq('').any():
        print('WARNING: missing plant/session IDs. An image-level estimate would not be scientifically independent; fill the manifest before grouped evaluation.')
    print(f'Saved {len(frame)} unique gate-passing rows to {output}')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=config.BASE_DIR / 'dataset_manifest.csv')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        build_dataset(args.manifest, config.DATASET_DIR, args.output or config.versioned_path('leaf_features', '.csv'))
    except (ValueError, OSError) as exc:
        print(f'Dataset build stopped: {exc}')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
