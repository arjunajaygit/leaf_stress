"""Quality checks and abstention use provisional thresholds, not validated semantics."""
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import app as web
import config
from feature_extraction import (analyze_leaf_full, check_mask_quality, morphology_kernel_size,
                                prepare_leaf, standardize_image)
from report_mask_quality import inspect_dataset, write_reports

ROOT = Path(__file__).resolve().parents[1]


def quality_for(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea) if contours else None
    selected = np.zeros_like(mask)
    if largest is not None:
        cv2.drawContours(selected, [largest], -1, 255, -1)
    return check_mask_quality(selected, largest, mask)


@pytest.mark.parametrize('size', [(1600, 1200), (256, 256), (400, 800), (1, 100)])
def test_working_resolution(size):
    image = np.zeros((*size, 3), dtype=np.uint8)
    resized = standardize_image(image)
    assert max(resized.shape[:2]) == 800
    assert abs(resized.shape[1] / resized.shape[0] - size[1] / size[0]) <= .13


def test_scaled_kernel():
    assert morphology_kernel_size((800, 600, 3)) == 5
    assert morphology_kernel_size((400, 300, 3)) == 3
    assert morphology_kernel_size((100, 100, 3)) == 1


def test_mask_empty():
    assert quality_for(np.zeros((800, 800), np.uint8))['reason_codes'] == ['no_leaf']


def test_full_frame_multiple_reasons():
    quality = quality_for(np.full((800, 800), 255, np.uint8))
    assert quality['reason_codes'] == ['border_contact', 'foreground_too_large']


def test_small_and_tiny():
    mask = np.zeros((800, 800), np.uint8)
    mask[100:110, 100:110] = 255
    assert quality_for(mask)['reason_codes'] == ['foreground_too_small', 'tiny_contour']


def test_clipped_contour():
    mask = np.zeros((800, 800), np.uint8)
    cv2.ellipse(mask, (0, 400), (100, 200), 0, 0, 360, 255, -1)
    assert quality_for(mask)['reason_codes'] == ['border_contact']


def test_components_before_largest_selection():
    mask = np.zeros((800, 800), np.uint8)
    cv2.ellipse(mask, (200, 400), (100, 200), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (600, 400), (80, 160), 0, 0, 360, 255, -1)
    quality = quality_for(mask)
    assert quality['components'] == quality['large_components'] == 2
    assert quality['reason_codes'] == ['multiple_components']


def test_small_speck_is_not_second_leaf():
    mask = np.zeros((800, 800), np.uint8)
    cv2.ellipse(mask, (400, 400), (100, 200), 0, 0, 360, 255, -1)
    mask[30:35, 30:35] = 255
    quality = quality_for(mask)
    assert quality['components'] == 2 and quality['large_components'] == 1
    assert quality['passed']


def test_threshold_boundaries(monkeypatch):
    mask = np.zeros((800, 800), np.uint8)
    mask[100:700, 100:700] = 255
    fraction = .5625
    monkeypatch.setattr(config, 'MASK_MAX_FOREGROUND_FRACTION', fraction)
    assert quality_for(mask)['passed']  # Strictly greater, not equal.
    monkeypatch.setattr(config, 'MASK_MAX_FOREGROUND_FRACTION', fraction - 1e-6)
    assert 'foreground_too_large' in quality_for(mask)['reason_codes']


def test_unreadable_pipeline(tmp_path):
    path = tmp_path / 'invalid.jpg'; path.write_bytes(b'bad')
    features, pictures, stats, quality = analyze_leaf_full(path)
    assert features is None and quality['reason_code'] == 'unreadable_image'


def test_gate_is_not_semantic_leaf_detection(tmp_path):
    # A green rectangular object can pass; explicitly preserve this known limitation.
    image = np.full((800, 800, 3), 255, np.uint8)
    image[200:600, 300:500] = (0, 150, 0)
    path = tmp_path / 'object.png'; cv2.imwrite(str(path), image)
    assert prepare_leaf(path)[-1]['passed']


def post_image(image):
    content = cv2.imencode('.png', image)[1].tobytes()
    return web.app.test_client().post('/predict', data={'image': (io.BytesIO(content), 'input.png')})


def leaf_image():
    image = np.full((800, 800, 3), 255, np.uint8)
    cv2.ellipse(image, (400, 400), (120, 230), 0, 0, 360, (0, 150, 0), -1)
    return image


class FakeModel:
    classes_ = ['Disease_Confounded', 'Healthy', 'Moderate_Water_Stress', 'Severe_Water_Stress']
    def __init__(self, probability=.8):
        self.probability = probability
        self.calls = 0
    def predict(self, X):
        self.calls += 1
        return ['Healthy']
    def predict_proba(self, X):
        p = self.probability
        return [[(1-p)/3, p, (1-p)/3, (1-p)/3]]


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(web, 'primary_model', model)
    monkeypatch.setattr(web, 'all_models', {})
    monkeypatch.setattr(web, 'bundle', {'preprocessing': config.preprocessing_signature()})
    monkeypatch.setitem(web.app.config, 'MIN_CLASS_PROBABILITY', .6)
    return model


@pytest.mark.parametrize('color', [(0, 150, 0), (0, 255, 255), (255, 255, 255)])
def test_rejection_skips_model_and_provides_preview(fake_model, color):
    response = post_image(np.full((800, 800, 3), color, np.uint8))
    assert response.status_code == 422
    body = response.get_json()
    assert body['outcome'] == 'rejected' and body['prediction'] is None
    assert body['visual_stress_indicator'] is None
    assert body['visualizations']['mask'].startswith('data:image/')
    assert body['visualizations']['candidate_mask'].startswith('data:image/')
    assert body['quality']['reason_codes']
    assert fake_model.calls == 0


@pytest.mark.parametrize('probability, expected', [(.5996, 'uncertain'), (.6, 'classified'), (.8, 'classified')])
def test_abstain_before_probability_rounding(fake_model, probability, expected):
    fake_model.probability = probability
    response = post_image(leaf_image())
    assert response.status_code == 200
    body = response.get_json()
    assert body['outcome'] == expected and body['threshold_provisional']
    if expected == 'uncertain':
        assert body['prediction'] is None and body['visual_stress_indicator'] is None
        assert 'multi_model_predictions' not in body
        assert body['top_probability'] == probability
        assert body['probabilities']['Healthy'] == round(probability, 3)
        assert set(body['probabilities']) == set(fake_model.classes_)
    else:
        assert body['prediction'] == 'Healthy'


def test_configurable_uncertainty_threshold(fake_model, monkeypatch):
    monkeypatch.setitem(web.app.config, 'MIN_CLASS_PROBABILITY', .9)
    assert post_image(leaf_image()).get_json()['outcome'] == 'uncertain'


def test_legacy_preprocessing_refused(fake_model, monkeypatch):
    monkeypatch.setattr(web, 'bundle', {})
    response = post_image(leaf_image())
    assert response.status_code == 503
    assert response.get_json()['code'] == 'preprocessing_mismatch'
    assert response.get_json()['prediction'] is None
    assert fake_model.calls == 0


@pytest.mark.parametrize('probability,expected',[(.4996,'uncertain'),(.5,'classified'),(.503,'classified'),(.6,'classified')])
def test_presentation_threshold_and_tentative_result(fake_model,monkeypatch,probability,expected):
    monkeypatch.setitem(web.app.config,'MIN_CLASS_PROBABILITY',.5)
    fake_model.probability=probability
    body=post_image(leaf_image()).get_json()
    assert body['outcome']==expected
    if expected=='classified':
        assert body['result_summary']['tentative']==(probability<.6)
        assert body['result_summary']['title']=='Low visible stress'
        assert 'soil' in body['result_summary']['next_check']
    else: assert body['prediction'] is None


def test_dataset_report_and_overlapping_counts(tmp_path):
    rows = inspect_dataset(config.DATASET_DIR)
    assert len(rows) == len({r['sha256'] for r in rows}) == 40
    summary = write_reports(rows, tmp_path / 'report.csv', tmp_path / 'summary.md')
    assert summary == {'pass': 11, 'fail': 29, 'reason_counts': {'border_contact': 29, 'foreground_too_large': 9}}
    saved = (ROOT / 'reports/mask_quality_report.csv').read_text()
    assert saved == (tmp_path / 'report.csv').read_text()
    with pytest.raises(FileExistsError):
        write_reports(rows, tmp_path / 'report.csv', tmp_path / 'summary.md')
