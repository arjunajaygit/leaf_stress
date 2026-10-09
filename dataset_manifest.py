"""Create an editable provenance manifest. Never infer biological/source metadata."""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import config
from feature_extraction import prepare_leaf

CLASSES = ('Healthy', 'Moderate_Water_Stress', 'Severe_Water_Stress', 'Disease_Confounded')
FEATURE_NAMES = ('h_mean', 'h_std', 's_mean', 's_std', 'v_mean', 'v_std', 'r_mean', 'g_mean',
                 'b_mean', 'exg_mean', 'gli_mean', 'solidity', 'aspect_ratio', 'extent', 'compactness')
HUMAN_FIELDS = ('plant_id', 'leaf_id', 'session_id', 'source', 'device', 'background_type', 'label_method', 'notes')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_paths(root):
    return [p for p in sorted(Path(root).glob('*/*')) if p.is_file() and p.suffix.lower() in config.ALLOWED_EXTENSIONS]


def create_manifest(root, output, stage3_report=None):
    root, output = Path(root), Path(output)
    if output.exists() or output.with_suffix('.metadata.json').exists():
        raise FileExistsError('Manifest already exists; use a new filename to avoid overwriting human annotations.')
    prior = {}
    if stage3_report:
        with Path(stage3_report).open(newline='') as stream:
            prior = {r['sha256']: r for r in csv.DictReader(stream)}
    seen, measured, rows = {}, {}, []
    for path in image_paths(root):
        digest = sha256(path)
        relative = path.relative_to(root).as_posix()
        if digest not in measured:
            *_, quality = prepare_leaf(path)
            image = cv2.imread(str(path))
            measured[digest] = (quality, image.shape[1] if image is not None else '', image.shape[0] if image is not None else '')
        quality, width, height = measured[digest]
        gate_status = 'pass' if quality['passed'] else 'fail'
        fraction = quality.get('foreground_fraction', '')
        if prior:
            old = prior.get(digest)
            if not old or old['pass_fail'] != gate_status or abs(float(old['foreground_fraction']) - fraction) > 1e-12:
                raise ValueError(f'Stage 3 snapshot mismatch for {relative}; do not silently change gate measurements.')
            gate_status, fraction = old['pass_fail'], old['foreground_fraction']
        row = {'filename': relative, 'class_label': path.parent.name, 'sha256': digest,
               'duplicate_of': seen.get(digest, ''), 'width': width, 'height': height,
               **{name: '' for name in HUMAN_FIELDS}, 'gate_status': gate_status,
               'border_contact_fraction': quality.get('border_contact_fraction', ''),
               'foreground_fraction': fraction, 'gate_reason_codes': ';'.join(quality['reason_codes'])}
        seen.setdefault(digest, relative)
        rows.append(row)
    if not rows:
        raise ValueError('No images found')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with output.with_suffix('.metadata.json').open('x') as stream:
        json.dump({'preprocessing': config.preprocessing_signature(), 'mask_quality': config.mask_quality_settings(),
                   'stage3_report': str(stage3_report) if stage3_report else None,
                   'stage3_report_sha256': sha256(stage3_report) if stage3_report else None,
                   'border_contact_fraction_definition': 'Sum of external contour segment lengths lying entirely on an image edge divided by total closed contour length; vertex-only contact contributes zero; absent/zero contour has fraction zero. Diagnostic only, not a gate threshold.',
                   'human_fields_left_blank': HUMAN_FIELDS}, stream, indent=2)
    return rows


def load_manifest(path, root):
    """Validate identity/path/hash fields, preserving human strings without guessing."""
    root = Path(root).resolve()
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    required = {'filename', 'class_label', 'sha256', *HUMAN_FIELDS}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('Manifest is empty or missing required fields')
    indexed = {}
    for row in rows:
        row = {k: (v or '').strip() for k, v in row.items()}
        relative = row['filename']; candidate = (root / relative).resolve()
        if Path(relative).is_absolute() or not candidate.is_relative_to(root) or relative in indexed:
            raise ValueError(f'Unsafe or repeated manifest filename: {relative}')
        if not candidate.is_file() or sha256(candidate) != row['sha256']:
            raise ValueError(f'Missing or changed image: {relative}; update and review the manifest')
        if row['class_label'] not in CLASSES or candidate.parent.name != row['class_label']:
            raise ValueError(f'Class/folder conflict for {relative}; resolve explicitly')
        indexed[relative] = row
    actual = {p.relative_to(root).as_posix() for p in image_paths(root)}
    if set(indexed) != actual:
        raise ValueError('Manifest does not cover exactly the dataset image files; create/reconcile a reviewed new manifest')
    return list(indexed.values())


def deduplicate(rows):
    """Fail on cross-label or identity conflicts instead of retaining a first label."""
    by_hash = {}
    for row in rows:
        by_hash.setdefault(row['sha256'], []).append(row)
    unique = []
    for digest, copies in by_hash.items():
        labels = {r['class_label'] for r in copies}
        if len(labels) != 1:
            raise ValueError('Cross-label duplicate conflict: ' + ', '.join(r['filename'] + '=' + r['class_label'] for r in copies))
        merged = dict(copies[0])
        for field in ('plant_id', 'leaf_id', 'session_id'):
            values = {r[field] for r in copies if r.get(field)}
            if len(values) > 1:
                raise ValueError(f'Conflicting {field} for identical image {digest}; review duplicate provenance')
            merged[field] = next(iter(values), '')
        merged['aliases'] = [r['filename'] for r in copies]
        unique.append(merged)
    return unique


def resolve_groups(rows):
    """Prefer plant IDs; use sessions conservatively for records missing plant IDs.

    Fallback session records are linked to any known plants in the same session.
    This avoids splitting an unknown-plant session copy from an identified plant.
    Fully identified different plants may share a session: plant grouping does not
    establish independence from session/environment effects.
    """
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(a, b):
        parent[find(b)] = find(a)
    plants, sessions, hashes = {}, {}, {}
    for i, row in enumerate(rows):
        if row.get('plant_id'):
            p = row['plant_id']
            if p in plants: union(i, plants[p])
            plants[p] = i
        if row.get('session_id'): sessions.setdefault(row['session_id'], []).append(i)
        if row['sha256'] in hashes: union(i, hashes[row['sha256']])
        hashes[row['sha256']] = i
    for indices in sessions.values():
        if any(not rows[i].get('plant_id') for i in indices):
            for i in indices[1:]: union(indices[0], i)
    components = {}
    for i in range(len(rows)): components.setdefault(find(i), []).append(i)
    groups = [('', 'missing')] * len(rows)
    for indices in components.values():
        ps = sorted({rows[i]['plant_id'] for i in indices if rows[i].get('plant_id')})
        ss = sorted({rows[i]['session_id'] for i in indices if rows[i].get('session_id')})
        if len(ps) == 1:
            identity, source = 'plant:' + ps[0], 'plant_id'
        elif ps:
            identity = 'linked:' + hashlib.sha256(json.dumps([ps, ss]).encode()).hexdigest()[:16]
            source = 'linked_plant_session'
        elif ss:
            identity, source = 'session:' + ss[0], 'session_id'
        else:
            continue
        for i in indices: groups[i] = (identity, source)
    return groups


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=config.BASE_DIR / 'dataset_manifest.csv')
    args = parser.parse_args()
    rows = create_manifest(config.DATASET_DIR, args.output, config.BASE_DIR / 'reports/mask_quality_report.csv')
    print(f'Created {len(rows)} rows. Fill plant_id, leaf_id, session_id, source, label_method, device and background_type; no metadata was inferred.')
