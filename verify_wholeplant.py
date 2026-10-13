"""Functional and local request timing evidence, never a held-out accuracy estimate."""
import io
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import time
from uuid import uuid4
import numpy as np


def main():
    root=Path(__file__).resolve().parent
    deployment=json.loads((root/'wholeplant_deployment_v1.json').read_text())
    os.environ['LECOR_ANALYSIS_MODE']='wholeplant'
    os.environ['LECOR_MODEL_PATH']=deployment['model']
    os.environ['LECOR_BENCHMARK_PATH']=deployment['report']
    os.environ['LECOR_PREVIEW_ONLY']='false'
    import app
    if app.primary_model is None: raise SystemExit(app.model_load_error)
    client=app.app.test_client();rows=[]
    for path in sorted((root/'dataset_wholeplant_v1_20261013').rglob('*.*')):
        start=time.perf_counter()
        response=client.post('/predict',data={'image':(io.BytesIO(path.read_bytes()),path.name)})
        elapsed=time.perf_counter()-start;data=response.get_json()
        if response.status_code!=200: raise RuntimeError((path,data))
        rows.append(dict(filename=path.relative_to(root).as_posix(),status=response.status_code,
                         outcome=data['outcome'],prediction=data['prediction'],seconds=elapsed))
    report=dict(date=datetime.now(timezone.utc).isoformat(),deployment=deployment,requests=len(rows),
        warning='Training images used for software verification, not independent prediction performance. Local Flask test client; excludes network/browser and model startup.',
        outcomes=dict(Counter(r['outcome'] for r in rows)), predicted_classes=dict(Counter(r['prediction'] for r in rows)),
        p50_seconds=float(np.percentile([r['seconds'] for r in rows],50)),
        p95_seconds=float(np.percentile([r['seconds'] for r in rows],95)),rows=rows)
    path=root/'reports'/('wholeplant_functional_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8]+'.json')
    path.write_text(json.dumps(report,indent=2));print(path);print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
