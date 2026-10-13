"""Real-checkpoint API smoke check; not a segmentation or classification benchmark."""
import base64
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import time
from uuid import uuid4


def main():
    root = Path(__file__).resolve().parent
    deployment = json.loads((root/'wholeplant_deployment_v1.json').read_text())
    os.environ['LECOR_ANALYSIS_MODE'] = 'wholeplant'
    os.environ['LECOR_MODEL_PATH'] = deployment['model']
    os.environ['LECOR_BENCHMARK_PATH'] = deployment['report']
    os.environ['LECOR_PREVIEW_ONLY'] = 'false'
    import app
    client = app.app.test_client()
    status = client.get('/api/sam/status').json
    if not status['available']:
        raise SystemExit(status['message'])
    # Known visible leaf and soil in this repository photo; no automatic IDs inferred.
    path = root/'dataset_wholeplant_v1_20261013/Healthy/healthy_001.JPG'
    points = [{'x': .27, 'y': .14, 'label': 1}, {'x': .92, 'y': .88, 'label': 0}]
    started = time.perf_counter()
    response = client.post('/api/sam/segment', data={'points': json.dumps(points),
        'image': (io.BytesIO(path.read_bytes()), path.name)})
    seconds = time.perf_counter() - started
    if response.status_code != 200:
        raise SystemExit(f'SAM failure: {response.status_code} {response.json}')
    data = response.json
    output = root/'reports'/('sam_smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8])
    output.mkdir(parents=True)
    for key in ('image', 'mask', 'overlay'):
        (output/f'{key}.png').write_bytes(base64.b64decode(data[key].split(',')[1]))
    prepared = (output/'image.png').read_bytes()
    preview = client.post('/predict', data={'image': (io.BytesIO(prepared), 'sam.png'),
        'mask_preview': 'true', 'input_preparation': 'sam'})
    assert preview.status_code == 200, preview.json
    prediction = client.post('/predict', data={'image': (io.BytesIO(prepared), 'sam.png'), 'input_preparation': 'sam'})
    report = dict(date=datetime.now(timezone.utc).isoformat(), source=str(path.relative_to(root)),
        points=points, status=status, segmentation_seconds=seconds,
        width=data['width'], height=data['height'], selected_fraction=data['selected_fraction'],
        preview_status=preview.status_code, preview_outcome=preview.json.get('outcome'),
        prediction_status=prediction.status_code, prediction_outcome=prediction.json.get('outcome'),
        model_load_error=app.model_load_error,
        warning='Software smoke test on a training photo with one prompted leaf. Not a test of complete-plant masks, tomato identity, segmentation quality or classifier accuracy.')
    (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2)); print('Evidence:', output)
    if prediction.status_code != 200:
        raise SystemExit('SAM and mask preview worked, but classification requires a compatible saved model.')


if __name__ == '__main__':
    main()
