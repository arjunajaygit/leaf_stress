"""Software acceptance checks; no plant-physiology or classifier-performance claim."""
import io
import os
from pathlib import Path
import subprocess
import sys
import cv2
import numpy as np
import pytest
import app as web
import config
from feature_extraction import (extract_color_features, extract_shape_features,
                                generate_visual_pipeline, prepare_leaf, analyze_leaf_full)
from model_contract import validate_bundle
from test_stage5 import fitted_bundle


def post(image):
    data = cv2.imencode('.png', image)[1].tobytes()
    return web.app.test_client().post('/predict', data={'image': (io.BytesIO(data), 'leaf.png')})


def leaf():
    image = np.full((200, 200, 3), 255, np.uint8)
    cv2.ellipse(image, (100, 100), (35, 65), 20, 0, 360, (30, 160, 50), -1)
    return image


@pytest.fixture
def preview(monkeypatch):
    monkeypatch.setitem(web.app.config, 'PREVIEW_ONLY', True)
    class Forbidden:
        def predict(self, X):
            pytest.fail('Preview must not call any model')
    monkeypatch.setattr(web, 'primary_model', Forbidden())
    monkeypatch.setattr(web, 'all_models', {'forbidden': Forbidden()})


def test_preview_skips_even_available_model_and_hides_benchmarks(preview):
    response = post(leaf()); body = response.get_json()
    assert response.status_code == 200 and body['outcome'] == 'preview_only'
    assert body['prediction'] is None and body['visual_stress_indicator'] is None
    assert 'probabilities' not in body and 'model_explanation' not in body
    assert body['visualizations']['mask'].startswith('data:image/')
    assert 'No validated classifier exists yet' in body['message']
    home = web.app.test_client().get('/').get_data(as_text=True)
    assert 'Preview-only demo' in home and 'Classification is unavailable' in home
    assert web.app.test_client().get('/api/benchmark').get_json()['available'] is False


def test_preview_rejections_and_repeatability(preview):
    assert post(leaf()).get_json() == post(leaf()).get_json()
    response = post(np.full((200, 200, 3), (0, 150, 0), np.uint8))
    assert response.status_code == 422
    body = response.get_json()
    assert body['outcome'] == 'rejected' and body['prediction'] is None
    assert 'border_contact' in body['quality']['reason_codes']


@pytest.mark.parametrize('preview_flag,should_start', [('true', True), ('false', False)])
def test_startup_without_model_in_subprocess(tmp_path, preview_flag, should_start):
    env = {**os.environ, 'LECOR_PREVIEW_ONLY': preview_flag, 'LECOR_MODEL_PATH': str(tmp_path/'missing.pkl')}
    code = 'import app; app.app.run=lambda **kw: print("SERVER_STARTED", kw); app.main()'
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
    assert (result.returncode == 0) == should_start
    if should_start:
        assert 'SERVER_STARTED' in result.stdout
        assert '127.0.0.1' in result.stdout and "'debug': False" in result.stdout
        assert 'Startup model error' not in result.stderr
    else:
        assert 'Startup error' in result.stderr and 'SERVER_STARTED' not in result.stdout


def test_shape_rectangle_circle_and_zero_perimeter():
    rectangle = np.array([[[0,0]], [[40,0]], [[40,20]], [[0,20]]], np.int32)
    shape = extract_shape_features(rectangle)
    assert shape['solidity'] == 1
    assert shape['extent'] == pytest.approx(800/(41*21))
    assert shape['aspect_ratio'] == pytest.approx(41/21)
    assert shape['compactness'] == pytest.approx(4*np.pi*800/120**2)
    points = np.linspace(0, 2*np.pi, 720, endpoint=False)
    circle = np.stack([50+20*np.cos(points), 50+20*np.sin(points)], axis=1).astype(np.float32).reshape(-1,1,2)
    assert extract_shape_features(circle)['compactness'] == pytest.approx(1, abs=.001)
    degenerate = extract_shape_features(np.array([[[1,1]]], np.int32))
    assert all(np.isfinite(v) for v in degenerate.values())
    assert degenerate['compactness'] == degenerate['solidity'] == 0


def test_rgb_order_gli_bounds_exg_identity_and_empty_mask():
    image = np.array([[[10,60,30], [0,0,0], [255,0,0], [0,255,0], [0,0,255]]], np.uint8)
    mask = np.full(image.shape[:2], 255, np.uint8)
    f = extract_color_features(image, mask)
    assert f['r_mean'] == pytest.approx(image[:,:,2].mean())
    assert f['b_mean'] == pytest.approx(image[:,:,0].mean())
    assert f['exg_mean'] == pytest.approx(2*f['g_mean']-f['r_mean']-f['b_mean'])
    for pixel in image[0]:
        f = extract_color_features(pixel.reshape(1,1,3), np.array([[255]], np.uint8))
        assert -1 <= f['gli_mean'] <= 1 and all(np.isfinite(v) for v in f.values())
    assert extract_color_features(image, np.zeros_like(mask)) is None


@pytest.mark.parametrize('color', [(0,180,0), (0,255,255), (0,0,0)])
@pytest.mark.parametrize('size', [1, 2, 10])
def test_tiny_color_visualizations_no_empty_overlay_crash(color, size):
    image = np.full((size, size, 3), color, np.uint8)
    mask = np.full((size, size), 255, np.uint8)
    contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    pictures, stats = generate_visual_pipeline(image, mask, contour)
    assert all(v.startswith('data:image/jpeg;base64,') for v in pictures.values())
    assert stats['total_pixels'] == size*size


@pytest.mark.parametrize('kind', ['blank', 'low_light', 'bright', 'complex', 'clipped', 'multiple'])
def test_segmentation_acquisition_failures(kind, tmp_path):
    image = leaf()
    if kind in ('blank', 'bright'): image[:] = 255
    elif kind == 'low_light': image[:] = (0, 10, 0)
    elif kind == 'complex': image[:] = (0, 150, 0)  # Green background merges with object.
    elif kind == 'clipped':
        image[:] = 255; cv2.ellipse(image, (0,100), (45,65), 0,0,360,(0,150,0),-1)
    else:
        image[:] = 255
        for x in (50,150): cv2.ellipse(image,(x,100),(25,50),0,0,360,(0,150,0),-1)
    path = tmp_path/'image.png'; cv2.imwrite(str(path),image)
    *_, features, quality = prepare_leaf(path)
    assert features is None and not quality['passed']


def test_truncated_image_json(preview):
    content = (config.DATASET_DIR/'Healthy/healthy_006.JPG').read_bytes()[:100]
    result = web.app.test_client().post('/predict', data={'image': (io.BytesIO(content), 'truncated.jpg')})
    assert result.status_code == 400 and result.is_json
    assert result.get_json()['code'] == 'invalid_image'


@pytest.mark.parametrize('mutation', ['extra_feature', 'estimator_order', 'comparison_class_order'])
def test_estimator_and_schema_mismatches(mutation):
    bundle, _ = fitted_bundle()
    if mutation == 'extra_feature': bundle['feature_schema'] += ['unknown']
    elif mutation == 'estimator_order': bundle['primary_model'][0].feature_names_in_ = np.array(list(reversed(bundle['feature_schema'])))
    else: bundle['primary_model'][-1].classes_ = bundle['primary_model'][-1].classes_[::-1]
    with pytest.raises(ValueError): validate_bundle(bundle)


def test_real_image_features_repeat_exactly():
    path = config.DATASET_DIR/'Healthy/healthy_006.JPG'
    first = analyze_leaf_full(path); second = analyze_leaf_full(path)
    assert first == second


def test_timing_rejects_invalid_repeats():
    from timing_preview import measure
    with pytest.raises(ValueError): measure(0)
