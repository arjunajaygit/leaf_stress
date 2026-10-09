"""Gate-passing nested grouped evaluation, or explicit legacy development-only diagnostics."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from build_dataset import prepare_dataset
from dataset_manifest import CLASSES, FEATURE_NAMES, sha256

OUTER_FOLDS = 3
INNER_FOLDS = 3
MIN_GROUPS_PER_CLASS = 5  # Protocol feasibility minimum, not statistical power.
SEED = 42
BLOCKED_MESSAGE = 'insufficient classes/groups for a credible generalization estimate'
LEGACY_WARNING = ('LEGACY DEVELOPMENT-ONLY: image-level numbers are NOT group-independent, NOT gate-checked, '
                  'and MUST NOT be reported as performance. They are historical diagnostic reproductions only. '
                  'The original features are incompatible with current resized preprocessing.')
TIE_BREAK = 'Full-precision mean inner-fold Macro F1, then mean balanced accuracy, then fixed model priority (LR, SVM, RF, KNN, GB), then candidate order.'


def candidate_models(include_baselines=False, legacy=False):
    def lr(c):
        return Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(C=c, max_iter=1000, random_state=SEED))])
    def svm(c):
        return Pipeline([('scaler', StandardScaler()), ('svc', SVC(C=c, kernel='rbf', random_state=SEED, **({'probability': True} if legacy else {})))])
    if legacy:
        candidates = [('Logistic Regression (L2)', lr(1.5), {'C': 1.5}),
                      ('Support Vector Machine (RBF)', svm(2.5), {'C': 2.5, 'gamma': 'scale'}),
                      ('Random Forest', RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_split=3, random_state=SEED), {'n_estimators': 250, 'max_depth': 8, 'min_samples_split': 3})]
    else:
        candidates = [('Logistic Regression (L2)', lr(c), {'C': c}) for c in (.3, 1.5)]
        candidates += [('Support Vector Machine (RBF)', svm(c), {'C': c, 'gamma': 'scale'}) for c in (.5, 2.5)]
        candidates += [('Random Forest', RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_leaf=leaf, random_state=SEED), {'n_estimators': 150, 'max_depth': 4, 'min_samples_leaf': leaf}) for leaf in (2, 4)]
    if include_baselines:
        candidates += [('K-Nearest Neighbors', Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))]), {'n_neighbors': 5, 'weights': 'distance'}),
                       ('Gradient Boosting', GradientBoostingClassifier(n_estimators=120 if legacy else 100, learning_rate=.08 if legacy else .05, max_depth=3 if legacy else 2, random_state=SEED), {'n_estimators': 120 if legacy else 100, 'learning_rate': .08 if legacy else .05, 'max_depth': 3 if legacy else 2})]
    return candidates


def metrics(y, predicted):
    y, predicted = np.asarray(y), np.asarray(predicted)
    report = classification_report(y, predicted, labels=list(CLASSES), output_dict=True, zero_division=0)
    healthy = y == 'Healthy'
    return {
        'accuracy': float(accuracy_score(y, predicted)),
        'macro_f1': float(f1_score(y, predicted, labels=list(CLASSES), average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y, predicted)),
        'macro_precision': float(report['macro avg']['precision']),
        'macro_recall': float(report['macro avg']['recall']),
        'weighted_f1': float(f1_score(y, predicted, labels=list(CLASSES), average='weighted', zero_division=0)),
        'per_class': {label: report[label] for label in CLASSES},
        'confusion_matrix_labels': list(CLASSES),
        'confusion_matrix': confusion_matrix(y, predicted, labels=list(CLASSES)).tolist(),
        'severe_to_healthy': int(np.sum((y == 'Severe_Water_Stress') & (predicted == 'Healthy'))),
        'disease_to_water_stress': int(np.sum((y == 'Disease_Confounded') & np.isin(predicted, ['Moderate_Water_Stress', 'Severe_Water_Stress']))),
        'healthy_false_alert_rate': float(np.mean(predicted[healthy] != 'Healthy')) if healthy.any() else None,
        'healthy_false_alert_definition': 'Any non-Healthy prediction among actual Healthy images, including Disease Confounded.',
    }


def readiness(frame):
    per_class = {}
    for label in CLASSES:
        subset = frame[frame['label'] == label]
        groups = set(subset['group_id']) - {''}
        unknown = int(subset['group_id'].eq('').sum())
        missing = []
        if not len(subset): missing.append('No gate-passing unique images')
        if len(subset) < MIN_GROUPS_PER_CLASS: missing.append(f'At least {MIN_GROUPS_PER_CLASS-len(subset)} more accepted images for protocol feasibility; independent groups still required')
        if unknown: missing.append(f'Fill plant_id or session_id for {unknown} accepted image(s)')
        if len(groups) < MIN_GROUPS_PER_CLASS: missing.append(f'At least {MIN_GROUPS_PER_CLASS-len(groups)} additional documented groups needed; currently unknown identities cannot be counted')
        per_class[label] = {'gate_passing_images': len(subset), 'documented_groups': len(groups),
                            'images_missing_group': unknown, 'missing': missing}
    return per_class


def validate_split(frame, train, test):
    train_rows, test_rows = frame.iloc[train], frame.iloc[test]
    if not len(train) or not len(test) or set(train) & set(test):
        raise ValueError('Invalid empty/overlapping fold')
    if set(train_rows.label) != set(CLASSES) or set(test_rows.label) != set(CLASSES):
        raise ValueError('A proposed fold lacks at least one class in training or validation')
    if set(train_rows.group_id) & set(test_rows.group_id):
        raise ValueError('Group leakage in proposed fold')
    if set(train_rows.sha256) & set(test_rows.sha256):
        raise ValueError('Duplicate hash leakage in proposed fold')


def nested_plan(frame):
    if any(item['missing'] for item in readiness(frame).values()):
        raise ValueError(BLOCKED_MESSAGE)
    if not frame['group_id'].ne('').all():
        raise ValueError('Missing group identifiers')
    outer = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
    plan = []
    for train, test in outer.split(frame, frame.label, frame.group_id):
        validate_split(frame, train, test)
        development = frame.iloc[train].reset_index(drop=True)
        inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
        folds = list(inner.split(development, development.label, development.group_id))
        for inner_train, inner_test in folds:
            validate_split(development, inner_train, inner_test)
        plan.append((train, test, folds))
    # Validate the entire plan before any fitting; never search seeds to obtain favorable folds.
    return plan


def selection_key(record):
    return (-record['macro_f1'], -record['balanced_accuracy'], record['priority'], record['candidate_index'])


def evaluate_nested(frame, candidates, feature_columns=FEATURE_NAMES):
    plan = nested_plan(frame)
    X, y = frame[list(feature_columns)], frame.label.to_numpy()
    names = list(dict.fromkeys(c[0] for c in candidates))
    per_model_predictions = {name: np.empty(len(frame), dtype=object) for name in names}
    selected_predictions = np.empty(len(frame), dtype=object)
    details = []
    for fold_number, (train, test, inner_folds) in enumerate(plan, 1):
        scores = []
        for index, (name, model, params) in enumerate(candidates):
            fold_metrics = []
            for fit, validation in inner_folds:
                fitted = clone(model).fit(X.iloc[train].iloc[fit], y[train][fit])
                predicted = fitted.predict(X.iloc[train].iloc[validation])
                fold_metrics.append(metrics(y[train][validation], predicted))
            scores.append({'model': name, 'parameters': params, 'candidate_index': index, 'priority': names.index(name),
                           'macro_f1': float(np.mean([m['macro_f1'] for m in fold_metrics])),
                           'balanced_accuracy': float(np.mean([m['balanced_accuracy'] for m in fold_metrics]))})
        winner = min(scores, key=selection_key)
        for name in names:
            best = min((s for s in scores if s['model'] == name), key=selection_key)
            fitted = clone(candidates[best['candidate_index']][1]).fit(X.iloc[train], y[train])
            predictions = fitted.predict(X.iloc[test])
            per_model_predictions[name][test] = predictions
            if best['candidate_index'] == winner['candidate_index']:
                selected_predictions[test] = predictions
        details.append({'fold': fold_number, 'selected_by_inner_cv': winner, 'inner_candidate_scores': scores,
                        'outer_train_indices': train.tolist(), 'outer_test_indices': test.tolist(),
                        'inner_indices_relative_to_outer_train': [{'train': a.tolist(), 'validation': b.tolist()} for a, b in inner_folds],
                        'selected_outer_metrics': metrics(y[test], selected_predictions[test])})
    return {'selected_procedure_metrics': metrics(y, selected_predictions),
            'model_comparisons': {name: metrics(y, p) for name, p in per_model_predictions.items()},
            'folds': details,
            'samples': [{'filename': r.filename, 'label': r.label, 'group_id': r.group_id,
                         'sha256': r.sha256, 'outer_prediction': str(p)} for r, p in zip(frame.itertuples(), selected_predictions)],
            'interpretation': 'Outer predictions estimate the complete inner model-selection procedure. Per-model outer comparisons are descriptive; choosing a winner using them would require a new independent final test. No production model is saved.'}


def grouped_evaluation(manifest, dataset_dir, include_baselines=False):
    frame, audit = prepare_dataset(manifest, dataset_dir)
    per_class = readiness(frame)
    warnings = ['Gate-passing images only; rejection coverage must accompany metrics.',
                'Group IDs depend on human provenance. Grouping does not validate labels or physiological claims.',
                'Five groups per class is a computational feasibility guard, not a statistically sufficient sample size.']
    if frame['group_id'].eq('').any():
        warnings.append('WARNING: missing plant/session identifiers. Only an image-level estimate would be possible, which is NOT scientifically independent. No automatic image-level fallback is run.')
    if frame['plant_id'].eq('').any() and frame['session_id'].ne('').any():
        warnings.append('Session fallback is used where plant IDs are unavailable. It cannot exclude the same unknown plant appearing across different sessions; plant independence is not established.')
    payload = {'mode': 'grouped_gate_passing', 'status': 'blocked', 'message': BLOCKED_MESSAGE,
               'warnings': warnings, 'per_class_readiness': per_class, 'audit': audit,
               'preprocessing': config.preprocessing_signature(), 'mask_quality': config.mask_quality_settings(),
               'protocol': {'outer_folds': OUTER_FOLDS, 'inner_folds': INNER_FOLDS, 'seed': SEED,
                            'minimum_documented_groups_per_class': MIN_GROUPS_PER_CLASS,
                            'selection': TIE_BREAK, 'metric_aggregation': 'Mean fold Macro F1 for inner selection; pooled outer predictions for the procedure estimate.'},
               'results': None}
    if any(item['missing'] for item in per_class.values()):
        return payload
    try:
        # All folds are preflighted before fitting; no automatic reduction/fabrication.
        nested_plan(frame)
    except ValueError as exc:
        payload['split_failure'] = str(exc)
        return payload
    payload['results'] = evaluate_nested(frame, candidate_models(include_baselines))
    payload.update(status='completed', message='Nested grouped CV completed; final external test and physiological validation remain outstanding.')
    return payload


def legacy_evaluation(include_baselines=False):
    path = config.BASE_DIR / 'leaf_features.csv'  # Deliberately original artifact, not the configured new CSV.
    expected = json.loads((config.BASE_DIR / 'backup_original/sha256_manifest.json').read_text())['leaf_features.csv']
    if sha256(path) != expected:
        raise ValueError('Original legacy CSV no longer matches the protected backup')
    frame = pd.read_csv(path)
    if len(frame) != 40 or set(frame.label) != set(CLASSES) or frame.filename.duplicated().any():
        raise ValueError('Expected the protected original 40-row four-class development CSV')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    for name, model, params in candidate_models(include_baselines, legacy=True):
        predicted = cross_val_predict(model, frame[list(FEATURE_NAMES)], frame.label, cv=cv)
        results[name] = {'parameters': params, **metrics(frame.label, predicted)}
    return {'mode': 'legacy_development_only_image_level', 'status': 'development_only', 'message': LEGACY_WARNING,
            'warnings': [LEGACY_WARNING, 'No tuning or production-model selection is performed. Original candidate settings are used for historical reproduction.'],
            'original_csv_sha256': sha256(path), 'samples': len(frame),
            'group_independent': False, 'gate_checked': False, 'report_as_performance': False,
            'documented_groups_per_class': {label: None for label in CLASSES},
            'results': results, 'protocol': {'folds': 5, 'seed': SEED, 'split': 'StratifiedKFold at image level', 'aggregation': 'Pooled out-of-fold diagnostic metrics'}}


def metric_markdown(title, data):
    lines = [f'### {title}', '', '| Metric | Value |', '|---|---:|']
    for metric in ('accuracy', 'macro_f1', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'weighted_f1', 'severe_to_healthy', 'disease_to_water_stress', 'healthy_false_alert_rate'):
        lines.append(f'| {metric} | {data[metric]} |')
    lines += ['', '| Class | Precision | Recall | F1 | Support |', '|---|---:|---:|---:|---:|']
    for label, row in data['per_class'].items():
        lines.append(f"| {label} | {row['precision']} | {row['recall']} | {row['f1-score']} | {row['support']} |")
    lines += ['', 'Confusion matrix (rows actual, columns predicted; order below):', '',
              '```json', json.dumps({'labels': data['confusion_matrix_labels'], 'matrix': data['confusion_matrix']}, indent=2), '```', '', data['healthy_false_alert_definition'], '']
    return lines


def write_outputs(payload, output_dir):
    when = datetime.now(ZoneInfo('Asia/Kolkata'))
    payload['generated_at'] = when.isoformat()
    prefix = 'legacy_development_only' if payload['mode'].startswith('legacy') else 'grouped_eval'
    stem = f"{prefix}_{when:%Y-%m-%d}_{when:%H%M%S}_{uuid4().hex[:8]}"
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = output_dir / (stem + '.json'), output_dir / (stem + '.md')
    with json_path.open('x') as stream: json.dump(payload, stream, indent=2, allow_nan=False)
    lines = ['# ' + ('LEGACY DEVELOPMENT-ONLY — NOT PERFORMANCE' if prefix.startswith('legacy') else 'Gate-passing grouped evaluation'), '', '**' + payload['message'] + '**', '',
             *['- ' + warning for warning in payload['warnings']], '', 'Generated: ' + payload['generated_at'], '']
    if 'per_class_readiness' in payload:
        lines += ['| Class | Gate-passing images | Documented groups | Missing information/data |', '|---|---:|---:|---|']
        for label, item in payload['per_class_readiness'].items():
            lines.append(f"| {label} | {item['gate_passing_images']} | {item['documented_groups']} | {'; '.join(item['missing']) or 'None for protocol feasibility'} |")
        lines += ['', f"Coverage: {sum(r['gate_passing_images'] for r in payload['per_class_readiness'].values())}/{payload['audit']['unique_images']} byte-unique images passed; {len(payload['audit']['rejected'])} rejected.", '',
                  'Group policy: plant_id first; session fallback links to identified plants in that session where necessary. Missing IDs are never filled with filenames as if they were independent plants.', '',
                  'Nested protocol: 3 outer folds and 3 inner folds, all classes required in every partition. At least 5 documented groups per class before split preflight; no invented folds or seed searching.', '', TIE_BREAK, '']
        if payload.get('split_failure'): lines += ['Split preflight failure: ' + payload['split_failure'], '']
    if payload['results'] is None:
        lines += [('**Split preflight failed; no models fitted and no evaluation numbers produced.**' if payload.get('split_failure') else '**No folds created, no models fitted and no evaluation numbers produced.**'), '']
    elif prefix.startswith('legacy'):
        lines += ['Groups per class: unknown. Gate status: not checked. Original 40 feature rows used unchanged.', '']
        for name, result in payload['results'].items(): lines += metric_markdown(name + ' — development-only diagnostics', result)
    else:
        lines += metric_markdown('Nested selection procedure', payload['results']['selected_procedure_metrics'])
        for name, result in payload['results']['model_comparisons'].items(): lines += metric_markdown(name + ' — nested tuned comparison', result)
        lines += [payload['results']['interpretation'], '', 'Exact sample identities, fold memberships and inner selections are recorded in the companion JSON.', '']
    with md_path.open('x') as stream: stream.write('\n'.join(lines))
    return json_path, md_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=config.BASE_DIR / 'dataset_manifest.csv')
    parser.add_argument('--output-dir', type=Path, default=config.BASE_DIR / 'reports')
    parser.add_argument('--legacy-development-only', action='store_true', help=LEGACY_WARNING)
    parser.add_argument('--include-baselines', action='store_true', help='Also include KNN and Gradient Boosting')
    args = parser.parse_args(argv)
    try:
        if args.legacy_development_only: print(LEGACY_WARNING)
        payload = legacy_evaluation(args.include_baselines) if args.legacy_development_only else grouped_evaluation(args.manifest, config.DATASET_DIR, args.include_baselines)
        paths = write_outputs(payload, args.output_dir)
    except (ValueError, OSError) as exc:
        print(f'Evaluation stopped: {exc}'); return 2
    print(payload['message'])
    for warning in payload['warnings']: print(warning)
    for label, item in payload.get('per_class_readiness', {}).items():
        print(f"{label}: accepted={item['gate_passing_images']}, groups={item['documented_groups']}; " + '; '.join(item['missing']))
    for path in paths: print(path)
    return 2 if payload['status'] == 'blocked' else 0


if __name__ == '__main__':
    raise SystemExit(main())
