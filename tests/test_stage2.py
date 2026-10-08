"""Claims and presentation contracts; these tests do not validate plant physiology."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import app as web
from feature_extraction import (
    compute_visual_stress_indicator, extract_all_features, generate_agronomic_advisory,
)

ROOT = Path(__file__).resolve().parents[1]
DISEASE_TEXT = 'Disease-like or other confounding symptoms suspected'


def test_indicator_integer_and_confounded_suppression():
    features = {'solidity': 1., 'compactness': 1., 'gli_mean': .4}
    result = compute_visual_stress_indicator(features, 'Healthy', {'Healthy': 1.})
    assert type(result) is int and 0 <= result <= 100
    assert compute_visual_stress_indicator(features, 'Disease_Confounded', {'Disease_Confounded': 1.}) is None


@pytest.mark.parametrize('label', ['Healthy', 'Moderate_Water_Stress', 'Severe_Water_Stress', 'Disease_Confounded', 'unknown'])
def test_guidance_asks_for_independent_checks(label):
    advice = generate_agronomic_advisory(label)
    text = ' '.join([advice['guidance'], *advice['action_items']])
    assert 'Check soil moisture directly' in text
    assert 'qualified agricultural expert' in text
    assert 'possible causes' in text
    for prohibited in ['L/m', 'fungicide', 'recovery_time', 'irrigation_plan', 'turgor loss detected']:
        assert prohibited not in str(advice)
    if label == 'Disease_Confounded':
        assert DISEASE_TEXT in advice['guidance']


@pytest.mark.parametrize('label', ['Healthy', 'Disease_Confounded'])
def test_prediction_contract(label, monkeypatch):
    class FixedModel:
        classes_ = web.classes
        def predict(self, X):
            return np.array([label])
        def predict_proba(self, X):
            return np.array([[float(c == label) for c in self.classes_]])
    monkeypatch.setattr(web, 'primary_model', FixedModel())
    monkeypatch.setattr(web, 'all_models', {})
    photo = (ROOT / 'dataset/Healthy/healthy_006.JPG').read_bytes()
    with web.app.test_client() as client:
        response = client.post('/predict', data={'image': (io.BytesIO(photo), 'leaf.JPG')})
    assert response.status_code == 200
    body = response.get_json()
    assert 'cwsi' not in body and 'xai_attributions' not in body
    assert len(body['feature_comparisons']) == 15
    if label == 'Disease_Confounded':
        assert body['visual_stress_indicator'] is None
    else:
        assert type(body['visual_stress_indicator']) is int


def test_comparisons_are_pooled_descriptions_not_importance(monkeypatch):
    monkeypatch.setattr(web, 'feature_names', ['solidity'])
    monkeypatch.setattr(web, 'feature_stats', {'solidity': {'mean': .8, 'std': .1}})
    monkeypatch.setattr(web, 'bundle', {'feature_importance': {'solidity': 9999}})
    comparison = web.compare_feature_measurements({'solidity': .6})[0]
    assert comparison['dataset_mean'] == .8
    assert comparison['z_score'] == -2.
    assert 'impact_score' not in comparison
    assert 'pooled dataset across all classes' in comparison['interpretation']
    assert 'not a healthy reference' in comparison['interpretation']


def test_production_features_unchanged():
    df = pd.read_csv(ROOT / 'leaf_features.csv')
    for row in df.to_dict('records'):
        features = extract_all_features(str(ROOT / 'dataset' / row['label'] / row['filename']))
        assert len(features) == 15
        for name, value in features.items():
            assert value == pytest.approx(row[name], abs=1e-9)


def test_visible_copy_and_srs_suggestions():
    with web.app.test_client() as client:
        text = client.get('/').get_data(as_text=True)
    for required in ['Experimental relative visual-stress indicator', 'Feature measurements vs. dataset averages',
                     'Predictions from other models on this image', 'Small dataset; visual-only assessment',
                     'not a physiological measurement', DISEASE_TEXT]:
        assert required in text
    for retired in ['CWSI', 'Otsu', 'Estimated Recovery', 'Irrigation Recommendation', 'Multi-Model Cross-Validation', 'Feature Attribution']:
        assert retired not in text
    doc = (ROOT / 'docs/srs_change_suggestions.md').read_text()
    for required in ['REQ-002', 'REQ-005', 'REQ-008', 'REQ-009', 'REQ-010', '1.3.4', 'Version 0.1', 'v0.3-4.pdf', 'does not yet check']:
        assert required in doc
