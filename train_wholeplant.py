"""Explicit development-only whole-plant training. No claim of independent validation."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import BASE_DIR
from dataset_manifest import CLASSES
from evaluate_grouped import metrics
from model_contract import library_versions, validate_bundle
import plant_features as pf

WARNING = ('EXPERIMENTAL DEVELOPMENT ONLY. Folder labels are unverified. Image-level CV is not plant/session-independent; '
           'near-duplicates and background/source confounding may inflate scores. Model selection uses the same CV, '
           'so the selected score is optimistic. No untouched test set or validated classifier exists. '
           'Do not present these numbers as real-world performance.')


def candidates():
    return {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(C=.3, max_iter=2000, random_state=42)),
        'RBF SVM': CalibratedClassifierCV(make_pipeline(StandardScaler(), SVC(C=1., random_state=42)), cv=3),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=2, random_state=42, n_jobs=1),
    }


def inventory(root):
    rows, features, seen, perceptual = [], [], {}, []
    for label in CLASSES:
        for path in sorted((root / label).glob('*')):
            if not path.is_file(): continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            relative = path.relative_to(root).as_posix()
            duplicate = seen.get(digest)
            if duplicate and duplicate[1] != label:
                raise ValueError(f'Cross-label duplicate conflict: {relative} and {duplicate[0]}')
            seen.setdefault(digest, (relative,label))
            row = dict(filename=relative, class_label=label, sha256=digest,
                duplicate_of=duplicate[0] if duplicate else '', width='', height='',
                plant_id='', leaf_id='', session_id='', source='', device='', background_type='', label_method='', notes='',
                gate_status='failed', reason_codes='', foreground_fraction='', border_contact='')
            try:
                with Image.open(path) as image:
                    row['width'], row['height'] = image.size
                    image.verify()
                image, mask, _, quality = pf.prepare_plant(path)
                row.update(gate_status='passed' if quality['passed'] else 'failed',
                    reason_codes=';'.join(quality['reason_codes']), foreground_fraction=quality['foreground_fraction'],
                    border_contact=quality['border_contact'])
                small = cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(9,8))
                perceptual.append((relative, (small[:,1:] > small[:,:-1]).flatten()))
                if quality['passed'] and not duplicate:
                    features.append(dict(filename=relative, label=label, sha256=digest, **pf.extract_features(image,mask)))
            except (ValueError, OSError) as exc:
                row['reason_codes'] = 'invalid_image'; row['notes'] = str(exc)
            rows.append(row)
    near = [dict(first=a, second=b, dhash_distance=int(np.count_nonzero(ah != bh)), status='candidate only; manual review required')
            for i,(a,ah) in enumerate(perceptual) for b,bh in perceptual[i+1:] if np.count_nonzero(ah != bh) <= 8]
    return rows, pd.DataFrame(features), near


def train(root, output, development_only=False):
    if not development_only:
        raise ValueError('Plant/session IDs and independent labels are unavailable. Pass --development-only to train an explicitly experimental demonstration model.')
    rows, frame, near = inventory(root)
    counts = frame['label'].value_counts() if len(frame) else {}
    missing = {label: int(counts.get(label,0)) for label in CLASSES if counts.get(label,0) < 5}
    if missing: raise ValueError(f'insufficient classes/groups for a credible generalization estimate; development training also requires 5 unique passing images/class: {missing}')
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output/'manifest.csv',index=False)
    frame.to_csv(output/'features.csv',index=False)
    X, y = frame[pf.FEATURE_NAMES], frame['label']
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(splitter.split(X,y))
    for train_ids, test_ids in splits:
        if set(frame.iloc[train_ids].sha256) & set(frame.iloc[test_ids].sha256): raise ValueError('Duplicate hash crosses folds')
    models, results = candidates(), {}
    for name, model in models.items():
        predicted = cross_val_predict(model, X, y, cv=splits)
        results[name] = metrics(y, predicted)
        results[name]['out_of_fold_predictions'] = [dict(filename=f, actual=a, predicted=p) for f,a,p in zip(frame.filename,y,predicted)]
        print(name, results[name]['macro_f1'], flush=True)
    # Full precision; simpler LR, then SVM, then RF wins an exact metric tie.
    names = list(models)
    best = max(names,key=lambda n:(results[n]['macro_f1'],results[n]['balanced_accuracy'],-names.index(n)))
    for model in models.values(): model.fit(X,y)
    protocol = dict(type='image-level development-only stratified 5-fold CV', independent=False,
        seed=42, selection='Macro F1, balanced accuracy, fixed LR/SVM/RF priority; full precision',
        warning=WARNING, test_set=None, groups_available=False, near_duplicate_candidates=near,
        grouping_status='insufficient classes/groups for a credible generalization estimate')
    signature = pf.preprocessing_signature()
    bundle = dict(analysis_mode='wholeplant', schema_version=pf.SCHEMA_VERSION, extractor_version=pf.EXTRACTOR_VERSION,
        preprocessing_version=signature['version'], preprocessing=signature, mask_quality=signature,
        feature_schema=pf.FEATURE_NAMES, feature_names_in_=pf.FEATURE_NAMES,
        dataset_hash=hashlib.sha256((output/'features.csv').read_bytes()).hexdigest(),
        dataset_hash_definition='SHA-256 of the exact training feature CSV bytes; rows include source image hashes',
        training_date=datetime.now(timezone.utc).isoformat(), library_versions=library_versions(), cv_protocol=protocol,
        classes_=list(models[best].classes_), primary_model=models[best], all_models=models, best_model_name=best,
        feature_stats={name:dict(mean=float(X[name].mean()), std=float(X[name].std(ddof=0))) for name in pf.FEATURE_NAMES})
    validate_bundle(bundle)
    joblib.dump(bundle, output/'model.joblib')
    report = dict(warning=WARNING, selected_model=best, cv_protocol=protocol, counts={k:int(v) for k,v in counts.items()},
        files=len(rows), accepted_unique=len(frame), features=len(pf.FEATURE_NAMES), results=results,
        rejected=[r for r in rows if r['gate_status']!='passed'], library_versions=library_versions(),
        missing_metadata='Fill plant_id, session_id, source, label_method, device and background_type in manifest. IDs must describe real identities, not invented row numbers.',
        dataset_source='https://drive.google.com/drive/folders/1JPmvvXO6p68wHXDrSWx2xE0-yXX7JUYt')
    (output/'development_report.json').write_text(json.dumps(report,indent=2))
    table='\n'.join(f'| {n} | {r["accuracy"]:.3f} | {r["macro_f1"]:.3f} | {r["balanced_accuracy"]:.3f} |' for n,r in results.items())
    (output/'development_report.md').write_text(f'# Whole-plant development diagnostics\n\n{WARNING}\n\n{len(rows)} files; {len(frame)} unique gate-passing images; counts: {report["counts"]}.\n\n| Candidate | Accuracy | Macro F1 | Balanced accuracy |\n|---|---:|---:|---:|\n{table}\n\nSelected development model: {best}. No independent test estimate.\n\nNear-duplicate candidates (not proof): {json.dumps(near)}\n\nFull per-class metrics, confusion matrices and out-of-fold predictions: development_report.json.\n\n{report["missing_metadata"]}\n')
    return output, report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=BASE_DIR/'dataset_wholeplant_v1_20261013')
    parser.add_argument('--development-only',action='store_true')
    args=parser.parse_args()
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8]
    output,report=train(args.dataset,BASE_DIR/'artifacts'/('wholeplant_v1_'+run_id),args.development_only)
    print(WARNING); print('Artifacts:',output); print('Selected:',report['selected_model'])


if __name__ == '__main__': main()
