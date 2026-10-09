"""Manifest integrity, grouped split feasibility and explicitly separated legacy mode."""
import csv
import importlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

import build_dataset as builder
import config
import dataset_manifest as manifest
import evaluate_grouped as evaluation
from feature_extraction import border_contact_fraction

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'dataset_manifest.csv'


def synthetic_frame(groups=6):
    # Synthetic software fixture: each group contains every class, not real evidence.
    rows = []
    for g in range(groups):
        for c, label in enumerate(manifest.CLASSES):
            rows.append({**{name: float(c + g*.01 + j*.1) for j, name in enumerate(manifest.FEATURE_NAMES)},
                         'label': label, 'filename': f'synthetic-{g}-{c}', 'sha256': f'hash-{g}-{c}',
                         'plant_id': f'p{g}', 'session_id': '', 'group_id': f'plant:p{g}', 'group_source': 'plant_id'})
    return pd.DataFrame(rows)


def test_manifest_counts_human_blanks_and_stage3_values():
    rows = manifest.load_manifest(MANIFEST, config.DATASET_DIR)
    assert len(rows) == 56 and len({r['sha256'] for r in rows}) == 40
    assert sum(bool(r['duplicate_of']) for r in rows) == 16
    assert all(not r[field] for r in rows for field in manifest.HUMAN_FIELDS)
    with (ROOT/'reports/mask_quality_report.csv').open(newline='') as f:
        prior = {r['sha256']: r for r in csv.DictReader(f)}
    for row in rows:
        assert row['gate_status'] == prior[row['sha256']]['pass_fail']
        assert float(row['foreground_fraction']) == float(prior[row['sha256']]['foreground_fraction'])
        assert 0 <= float(row['border_contact_fraction']) <= 1
    metadata = json.loads(MANIFEST.with_suffix('.metadata.json').read_text())
    assert metadata['mask_quality'] == config.mask_quality_settings()
    assert metadata['preprocessing'] == config.preprocessing_signature()


def test_border_perimeter_fraction():
    interior = np.array([[[1,1]], [[9,1]], [[9,9]], [[1,9]]], dtype=np.int32)
    touching_one_edge = np.array([[[0,1]], [[8,1]], [[8,9]], [[0,9]]], dtype=np.int32)
    full_frame = np.array([[[0,0]], [[9,0]], [[9,9]], [[0,9]]], dtype=np.int32)
    vertex_only = np.array([[[0,5]], [[5,1]], [[9,5]], [[5,9]]], dtype=np.int32)
    assert border_contact_fraction(interior, (11,11)) == 0
    assert border_contact_fraction(touching_one_edge, (11,11)) == .25
    assert border_contact_fraction(full_frame, (10,10)) == 1
    assert border_contact_fraction(vertex_only, (11,11)) == 0
    assert border_contact_fraction(None, (10,10)) == 0
    assert border_contact_fraction(np.array([[[0,0]]]), (10,10)) == 0


def test_manifest_generation_does_not_overwrite(tmp_path):
    out = tmp_path/'manifest.csv'
    rows = manifest.create_manifest(config.DATASET_DIR, out, ROOT/'reports/mask_quality_report.csv')
    assert len(rows) == 56
    with pytest.raises(FileExistsError):
        manifest.create_manifest(config.DATASET_DIR, out)


def test_import_builder_has_no_writes(tmp_path):
    result = subprocess.run([sys.executable, '-c',
        f"import sys; sys.path.insert(0, {str(ROOT)!r}); import config; from pathlib import Path; config.BASE_DIR=Path({str(tmp_path)!r}); import build_dataset"],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


def row(name, label='Healthy', digest=None, plant='', session='', leaf=''):
    return {'filename': name, 'class_label': label, 'sha256': digest or name,
            'plant_id': plant, 'session_id': session, 'leaf_id': leaf}


def test_cross_label_duplicates_fail():
    with pytest.raises(ValueError, match='Cross-label duplicate conflict'):
        manifest.deduplicate([row('a', digest='same'), row('b', label='Severe_Water_Stress', digest='same')])


def test_duplicate_identity_conflict_and_blanks():
    copies = [row('a', digest='same'), row('b', digest='same', plant='P1')]
    unique = manifest.deduplicate(copies)
    assert len(unique) == 1 and unique[0]['plant_id'] == 'P1'
    with pytest.raises(ValueError, match='Conflicting plant_id'):
        manifest.deduplicate([row('a', digest='same', plant='P1'), row('b', digest='same', plant='P2')])


def test_plant_session_fallback_and_namespace():
    rows = [row('a', plant='P1', session='S1'), row('b', plant='P1', session='S2'),
            row('c', session='S1'), row('d', session='S3'), row('e'), row('f', plant='S3')]
    groups = manifest.resolve_groups(rows)
    assert groups[0] == groups[1] == groups[2] == ('plant:P1', 'plant_id')
    assert groups[3] == ('session:S3', 'session_id')
    assert groups[4] == ('', 'missing')
    assert groups[5][0] != groups[3][0]


def test_fallback_session_links_multiple_plants_conservatively():
    rows = [row('a', plant='P1', session='S1'), row('b', plant='P2', session='S1'), row('c', session='S1')]
    groups = manifest.resolve_groups(rows)
    assert groups[0] == groups[1] == groups[2]
    assert groups[0][1] == 'linked_plant_session'


def test_current_dataset_stops_before_folds_or_fits(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('No folds/models may be created for this incomplete dataset')
    monkeypatch.setattr(evaluation, 'nested_plan', forbidden)
    monkeypatch.setattr(evaluation, 'candidate_models', forbidden)
    result = evaluation.grouped_evaluation(MANIFEST, config.DATASET_DIR)
    assert result['status'] == 'blocked' and result['results'] is None
    assert result['message'] == 'insufficient classes/groups for a credible generalization estimate'
    counts = {k: v['gate_passing_images'] for k,v in result['per_class_readiness'].items()}
    assert counts == {'Healthy':4, 'Moderate_Water_Stress':0, 'Severe_Water_Stress':0, 'Disease_Confounded':7}
    assert all(r['documented_groups'] == 0 for r in result['per_class_readiness'].values())
    assert any('NOT scientifically independent' in w for w in result['warnings'])


def test_manifest_gate_status_cannot_bypass_gate(tmp_path):
    frame = pd.read_csv(MANIFEST, keep_default_na=False)
    frame['gate_status'] = 'pass'
    path = tmp_path/'manifest.csv'; frame.to_csv(path, index=False)
    accepted, audit = builder.prepare_dataset(path, config.DATASET_DIR)
    assert len(accepted) == 11 and len(audit['rejected']) == 29


def test_builder_reads_identity_fields(tmp_path):
    frame = pd.read_csv(MANIFEST, keep_default_na=False)
    frame['plant_id'] = 'real-id-provided-in-test-fixture'
    path = tmp_path/'manifest.csv'; frame.to_csv(path, index=False)
    accepted, _ = builder.prepare_dataset(path, config.DATASET_DIR)
    assert accepted.group_id.eq('plant:real-id-provided-in-test-fixture').all()


def test_builder_fails_cleanly_without_partial_csv(tmp_path):
    path = tmp_path/'features_v1_test.csv'
    with pytest.raises(ValueError, match='Too few gate-passing samples'):
        builder.build_dataset(MANIFEST, config.DATASET_DIR, path)
    assert not path.exists() and not path.with_suffix('.metadata.json').exists()
    assert builder.main(['--output', str(path)]) == 2


def test_successful_builder_uses_groups_and_sidecar(tmp_path, monkeypatch):
    frame = synthetic_frame()
    monkeypatch.setattr(builder, 'prepare_dataset', lambda *args: (frame, {'synthetic_test_fixture': True}))
    path = tmp_path/'features_v1_test.csv'
    builder.build_dataset(MANIFEST, config.DATASET_DIR, path)
    sidecar = json.loads(path.with_suffix('.metadata.json').read_text())
    assert sidecar['groups_available']
    assert sidecar['csv_sha256'] == manifest.sha256(path)
    assert set(pd.read_csv(path).group_id) == set(frame.group_id)
    with pytest.raises(FileExistsError): builder.build_dataset(MANIFEST, config.DATASET_DIR, path)


def test_group_and_duplicate_separation_in_every_nested_fold():
    frame = synthetic_frame()
    plan = evaluation.nested_plan(frame)
    tested = []
    for train, test, inner in plan:
        evaluation.validate_split(frame, train, test)
        assert not set(frame.iloc[train].group_id) & set(frame.iloc[test].group_id)
        tested.extend(test)
        dev = frame.iloc[train].reset_index(drop=True)
        for fit, validation in inner: evaluation.validate_split(dev, fit, validation)
    assert sorted(tested) == list(range(len(frame)))


def test_explicit_duplicate_leakage_detection():
    frame = synthetic_frame()
    train, test, _ = evaluation.nested_plan(frame)[0]
    frame.loc[test[0], 'sha256'] = frame.loc[train[0], 'sha256']
    with pytest.raises(ValueError, match='Duplicate hash leakage'):
        evaluation.validate_split(frame, train, test)


def test_missing_group_blocks_no_image_fallback():
    frame = synthetic_frame(); frame.loc[0,'group_id'] = ''
    with pytest.raises(ValueError, match='insufficient classes/groups'):
        evaluation.nested_plan(frame)


def test_nested_success_runs_selection_and_is_reproducible():
    frame = synthetic_frame()
    candidates = evaluation.candidate_models()[:3]  # LR alternatives and an RBF SVM.
    a = evaluation.evaluate_nested(frame, candidates)
    b = evaluation.evaluate_nested(frame, candidates)
    assert a == b
    assert len(a['folds']) == 3 and len(a['samples']) == len(frame)
    assert a['selected_procedure_metrics']['confusion_matrix_labels'] == list(manifest.CLASSES)
    assert set(a['model_comparisons']) == {'Logistic Regression (L2)', 'Support Vector Machine (RBF)'}


def test_full_precision_tie_break():
    a = {'macro_f1': .60001, 'balanced_accuracy': .7, 'priority': 0, 'candidate_index': 0}
    b = {'macro_f1': .60002, 'balanced_accuracy': .6, 'priority': 1, 'candidate_index': 1}
    assert min([a,b], key=evaluation.selection_key) is b
    b.update(macro_f1=a['macro_f1'], balanced_accuracy=a['balanced_accuracy'])
    assert min([b,a], key=evaluation.selection_key) is a


def test_candidates_scaling_and_rf_regularization():
    for name, model, _ in evaluation.candidate_models(include_baselines=True):
        if name in ('Logistic Regression (L2)', 'Support Vector Machine (RBF)', 'K-Nearest Neighbors'):
            assert isinstance(model, Pipeline) and 'scaler' in model.named_steps
        if name == 'Random Forest':
            assert model.max_depth == 4 and model.min_samples_leaf >= 2


def test_error_metrics_definition():
    y = ['Healthy','Healthy','Moderate_Water_Stress','Severe_Water_Stress','Disease_Confounded','Disease_Confounded']
    p = ['Disease_Confounded','Healthy','Moderate_Water_Stress','Healthy','Severe_Water_Stress','Healthy']
    result = evaluation.metrics(y,p)
    assert result['severe_to_healthy'] == 1
    assert result['disease_to_water_stress'] == 1  # Disease→Healthy is not counted here.
    assert result['healthy_false_alert_rate'] == .5


def test_blocked_reports_and_legacy_filenames_separate(tmp_path):
    grouped = evaluation.grouped_evaluation(MANIFEST, config.DATASET_DIR)
    paths = evaluation.write_outputs(grouped, tmp_path)
    assert paths[0].name.startswith('grouped_eval_')
    assert 'No folds created, no models fitted' in paths[1].read_text()
    legacy = {'mode':'legacy_development_only_image_level','status':'development_only',
              'message':evaluation.LEGACY_WARNING,'warnings':[evaluation.LEGACY_WARNING],'results':{}}
    old_paths = evaluation.write_outputs(legacy, tmp_path)
    assert old_paths[0].name.startswith('legacy_development_only_')
    assert 'MUST NOT be reported as performance' in old_paths[1].read_text()
    assert not set(paths) & set(old_paths)


def test_legacy_numbers_reproduce_protected_original():
    result = evaluation.legacy_evaluation(include_baselines=True)
    original = json.loads((ROOT/'model_benchmark.json').read_text())
    assert result['samples'] == 40 and result['report_as_performance'] is False
    assert result['group_independent'] is False and result['gate_checked'] is False
    for name, numbers in result['results'].items():
        assert numbers['accuracy'] == original['models'][name]['accuracy']
        assert numbers['macro_f1'] == pytest.approx(original['models'][name]['macro_f1'], abs=0.00005)
