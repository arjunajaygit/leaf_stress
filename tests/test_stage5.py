import copy
import hashlib
import json
import warnings
import joblib
import cv2
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import app as web
import config
import ablate_features as ablation
from dataset_manifest import FEATURE_NAMES, CLASSES
from model_contract import training_metadata, validate_bundle, explain_logistic
from feature_extraction import extract_experimental_features


def fitted_bundle():
    frame = ablation.synthetic_frame()
    model = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000))]).fit(frame[list(FEATURE_NAMES)], frame.label)
    return {**training_metadata('a'*64, {'split': 'synthetic grouped fixture; not performance'}),
            'preprocessing': config.preprocessing_signature(), 'mask_quality': config.mask_quality_settings(),
            'primary_model': model, 'all_models': {'LR': model}, 'classes_': list(model.classes_),
            'feature_names_in_': list(FEATURE_NAMES)}, frame


def test_original_load_has_no_numpy_joblib_warnings():
    # This assertion targets the pinned environment, not an unrelated global install.
    if np.__version__ != '2.4.6':
        pytest.skip('Run with the pinned NumPy 2.4.6 environment')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        joblib.load(config.BASE_DIR / 'leaf_stress_model.pkl')
    assert not caught


def test_new_bundle_round_trip(tmp_path, monkeypatch):
    bundle, _ = fitted_bundle()
    validate_bundle(bundle)
    path = tmp_path / 'model_v1_test.pkl'
    joblib.dump(bundle, path)
    for name in ['bundle', 'primary_model', 'all_models', 'feature_names', 'classes', 'feature_stats', 'model_load_error']:
        monkeypatch.setattr(web, name, getattr(web, name))
    assert web.load_model_bundle(path)
    assert web.primary_model is not None


@pytest.mark.parametrize('field,value', [
    ('schema_version', 'wrong'), ('extractor_version', 'wrong'), ('preprocessing_version', 'wrong'),
    ('feature_schema', list(reversed(FEATURE_NAMES))), ('feature_names_in_', list(FEATURE_NAMES)[:-1]),
    ('dataset_hash', 'unknown'), ('library_versions', {}), ('training_date', ''), ('cv_protocol', {}),
    ('classes_', list(reversed(sorted(CLASSES)))), ('mask_quality', {})])
def test_contract_rejects_mismatch(field, value, tmp_path, monkeypatch):
    bundle, _ = fitted_bundle(); bundle[field] = value
    with pytest.raises(ValueError): validate_bundle(bundle)
    path = tmp_path / 'bad.pkl'; joblib.dump(bundle, path)
    for name in ['bundle', 'primary_model', 'all_models', 'feature_names', 'classes', 'feature_stats', 'model_load_error']:
        monkeypatch.setattr(web, name, getattr(web, name))
    assert not web.load_model_bundle(path)
    assert web.primary_model is None and web.bundle is None


def test_original_refused_at_load(monkeypatch):
    for name in ['bundle', 'primary_model', 'all_models', 'feature_names', 'classes', 'feature_stats', 'model_load_error']:
        monkeypatch.setattr(web, name, getattr(web, name))
    assert not web.load_model_bundle()
    assert 'schema_version' in web.model_load_error
    assert web.primary_model is None


def test_exact_logistic_explanation():
    bundle, frame = fitted_bundle(); model = bundle['primary_model']
    for label in model.classes_:
        X = frame[frame.label == label][list(FEATURE_NAMES)].iloc[:1]
        predicted = model.predict(X)[0]
        result = explain_logistic(model, X, predicted)
        assert result['available']
        assert np.isclose(sum(r['contribution'] for r in result['contributions']) + result['intercept'], result['decision_score'])
        index = list(model.classes_).index(predicted)
        np.testing.assert_allclose([r['contribution'] for r in result['contributions']], model[-1].coef_[index] * model[0].transform(X)[0])
    assert not explain_logistic(object(), X, predicted)['available']
    with pytest.raises(ValueError): explain_logistic(model, X[list(reversed(FEATURE_NAMES))], predicted)


def test_experimental_feature_math():
    image = np.full((20, 30, 3), (10, 60, 30), np.uint8); mask = np.full((20, 30), 255, np.uint8)
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    result = extract_experimental_features(image, mask, contour)
    assert result['chrom_r_mean'] == pytest.approx(.3)
    assert result['chrom_g_mean'] == pytest.approx(.6)
    assert result['oriented_axis_ratio'] == pytest.approx(19/29)
    expected = cv2.cvtColor(image.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
    assert result['lab_a_mean'] == pytest.approx(expected[0,0,1])
    assert result['lab_b_std'] == 0
    assert all(np.isfinite(v) for v in result.values())
    assert extract_experimental_features(np.zeros_like(image), mask, contour)['chrom_r_mean'] == 0
    with pytest.raises(ValueError): extract_experimental_features(image, np.zeros_like(mask), contour)


def test_ablation_nested_groups_and_identical_folds():
    frame = ablation.synthetic_frame()
    model = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=1000))])
    results = ablation.run_ablation(frame, [('LR', model, {})])
    assert list(results) == list(ablation.VARIANTS)
    reference = results['baseline_15']['folds']
    for result in results.values():
        for fold, original in zip(result['folds'], reference):
            assert fold['outer_test_indices'] == original['outer_test_indices']
            train, test = frame.iloc[fold['outer_train_indices']], frame.iloc[fold['outer_test_indices']]
            assert not set(train.group_id) & set(test.group_id)
            for inner in fold['inner_indices_relative_to_outer_train']:
                assert not set(train.iloc[inner['train']].group_id) & set(train.iloc[inner['validation']].group_id)
    assert len(ablation.VARIANTS['baseline_15']) == 15
    assert 'exg_mean' not in ablation.VARIANTS['without_exg']
    assert 'aspect_ratio' not in ablation.VARIANTS['oriented_ratio']


def test_real_ablation_blocked_before_any_fit(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ablation, 'candidate_models', lambda: pytest.fail('Real dataset must not fit models'))
    # run_ablation currently evaluates candidates only after all split preflight checks.
    assert ablation.main(['--output-dir', str(tmp_path)]) == 2
    assert ablation.BLOCKED_MESSAGE in capsys.readouterr().out
    payload = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert payload['results'] is None
    assert payload['per_class_readiness']['Moderate_Water_Stress']['gate_passing_images'] == 0


def test_training_writer_persists_contract_without_touching_production(tmp_path, monkeypatch):
    import train_model
    from model_contract import SCHEMA_VERSION, EXTRACTOR_VERSION
    frame = ablation.synthetic_frame()[[*FEATURE_NAMES, 'label', 'filename']]
    csv = tmp_path / 'features_v1_synthetic.csv'; frame.to_csv(csv, index=False)
    metadata = {'preprocessing': config.preprocessing_signature(), 'mask_quality': config.mask_quality_settings(),
                'schema_version': SCHEMA_VERSION, 'extractor_version': EXTRACTOR_VERSION,
                'preprocessing_version': config.preprocessing_signature()['version'],
                'feature_names': list(FEATURE_NAMES), 'csv_sha256': hashlib.sha256(csv.read_bytes()).hexdigest()}
    csv.with_suffix('.metadata.json').write_text(json.dumps(metadata))
    monkeypatch.setattr(train_model, 'CSV_PATH', csv)
    monkeypatch.setattr(train_model, 'versioned_path', lambda stem, suffix: tmp_path / (stem + '_v1_synthetic' + suffix))
    train_model.run_benchmarks()
    saved = joblib.load(tmp_path / 'leaf_stress_model_v1_synthetic.pkl')
    validate_bundle(saved)
    assert saved['dataset_hash'] == metadata['csv_sha256']
    assert saved['cv_protocol']['group_independent'] is False
    assert saved['feature_schema'] == list(FEATURE_NAMES)


def test_api_includes_faithful_explanation(tmp_path, monkeypatch):
    import io
    bundle, _ = fitted_bundle()
    path = tmp_path / 'new_v1.pkl'; joblib.dump(bundle, path)
    for name in ['bundle', 'primary_model', 'all_models', 'feature_names', 'classes', 'feature_stats', 'model_load_error']:
        monkeypatch.setattr(web, name, getattr(web, name))
    monkeypatch.setitem(web.app.config, 'MIN_CLASS_PROBABILITY', 0.)
    assert web.load_model_bundle(path)
    photo = (config.DATASET_DIR / 'Healthy/healthy_006.JPG').read_bytes()
    response = web.app.test_client().post('/predict', data={'image': (io.BytesIO(photo), 'leaf.JPG')})
    assert response.status_code == 200
    body = response.get_json()
    assert body['model_explanation']['available']
    assert body['model_explanation']['predicted_class'] == body['prediction']
