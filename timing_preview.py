"""REQ-013 preview endpoint timing, not classification latency or browser/network timing."""
import argparse
from collections import Counter
from datetime import datetime
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import platform
from time import perf_counter
from uuid import uuid4
from zoneinfo import ZoneInfo
import numpy as np
import config
from dataset_manifest import image_paths, sha256


def measure(repeats=3):
    if repeats < 1:
        raise ValueError('repeats must be positive')
    # Set before import; no model is deserialized or used by this script.
    old_env = os.environ.get('LECOR_PREVIEW_ONLY')
    old_config = config.PREVIEW_ONLY
    os.environ['LECOR_PREVIEW_ONLY'] = 'true'
    config.PREVIEW_ONLY = True
    import app as web
    old = web.app.config['PREVIEW_ONLY']
    web.app.config['PREVIEW_ONLY'] = True
    try:
        seen, photos = set(), []
        for path in image_paths(config.DATASET_DIR):
            digest = sha256(path)
            if digest not in seen:
                photos.append(path); seen.add(digest)
        if not photos:
            raise ValueError('No dataset images found')
        client = web.app.test_client()
        def post(path, data):
            return client.post('/predict', data={'image': (io.BytesIO(data), path.name)})
        # One warm-up excluded. Each measured request includes upload serialization,
        # decode, resizing, segmentation, visualizations, JSON and temporary cleanup.
        post(photos[0], photos[0].read_bytes())
        rows = []
        for repeat in range(repeats):
            for path in photos:
                data = path.read_bytes()  # Exclude dataset disk read, like an already-selected upload.
                start = perf_counter(); response = post(path, data); elapsed = perf_counter() - start
                body = response.get_json()
                if response.status_code not in (200, 422) or body.get('outcome') not in ('preview_only', 'rejected'):
                    raise RuntimeError(f'Unexpected timing response: {response.status_code}')
                rows.append({'filename': path.relative_to(config.DATASET_DIR).as_posix(), 'repeat': repeat+1,
                             'seconds': elapsed, 'status': response.status_code, 'outcome': body['outcome']})
        durations = [r['seconds'] for r in rows]
        return {'mode': 'preview_only', 'classification_measured': False,
                'limitation': 'In-process Flask client on this machine, one warm-up excluded. No network/browser/classification. Does not verify REQ-013 classification within 10 seconds.',
                'python': platform.python_version(), 'platform': platform.platform(),
                'numpy': version('numpy'), 'unique_images': len(photos), 'repeats': repeats, 'requests': len(rows),
                'p50_seconds': float(np.percentile(durations, 50)), 'p95_seconds': float(np.percentile(durations, 95)),
                'max_seconds': float(max(durations)), 'outcome_counts': dict(Counter(r['outcome'] for r in rows)), 'samples': rows}
    finally:
        web.app.config['PREVIEW_ONLY'] = old
        config.PREVIEW_ONLY = old_config
        if old_env is None:
            os.environ.pop('LECOR_PREVIEW_ONLY', None)
        else:
            os.environ['LECOR_PREVIEW_ONLY'] = old_env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output-dir', type=Path, default=config.BASE_DIR / 'reports')
    args = parser.parse_args(argv)
    result = measure(args.repeats)
    when = datetime.now(ZoneInfo('Asia/Kolkata')); result['generated_at'] = when.isoformat()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f'preview_timing_{when:%Y-%m-%d_%H%M%S}_{uuid4().hex[:8]}.json'
    with path.open('x') as stream: json.dump(result, stream, indent=2)
    print(f"Preview only: p50={result['p50_seconds']:.4f}s, p95={result['p95_seconds']:.4f}s; classification latency NOT measured.")
    print(path)


if __name__ == '__main__':
    main()
