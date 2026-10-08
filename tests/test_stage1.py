"""Regression and security checks for Stage 1. Never writes original artifacts."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

import app as web
import config
from feature_extraction import analyze_leaf_full, generate_visual_pipeline

ROOT = Path(__file__).resolve().parents[1]
HEALTHY = ROOT / 'dataset' / 'Healthy'


@pytest.fixture(autouse=True)
def isolate_globals(monkeypatch):
    for name in ['bundle', 'primary_model', 'all_models', 'feature_names', 'classes', 'feature_stats', 'model_load_error']:
        monkeypatch.setattr(web, name, getattr(web, name))
    # Test-only compatibility marker to exercise old API regression paths.
    # Production legacy artifacts are explicitly refused by Stage 3 tests.
    monkeypatch.setattr(web, 'bundle', {**(web.bundle or {}), 'preprocessing': config.preprocessing_signature()})
    monkeypatch.setitem(web.app.config, 'MIN_CLASS_PROBABILITY', 0.0)
    monkeypatch.setitem(web.app.config, 'TESTING', True)
    monkeypatch.setitem(web.app.config, 'MAX_CONTENT_LENGTH', 10 * 1024 * 1024)
    monkeypatch.setitem(web.app.config, 'MAX_IMAGE_PIXELS', 16_000_000)


@pytest.fixture
def client():
    return web.app.test_client()


def upload(client, data=None, name='leaf.JPG'):
    data = data if data is not None else (HEALTHY / 'healthy_006.JPG').read_bytes()
    return client.post('/predict', data={'image': (io.BytesIO(data), name)}, content_type='multipart/form-data')


def assert_error(response, status, code):
    assert response.status_code == status
    assert response.is_json
    body = response.get_json()
    assert body['code'] == code
    assert isinstance(body['error'], str) and body['error']


@pytest.mark.parametrize('name', ['healthy_006.JPG', 'healthy_008.JPG'])
def test_healthy_visualization_regression(name):
    features, pictures, stats, quality = analyze_leaf_full(str(HEALTHY / name))
    assert quality["passed"]
    assert len(features) == 15
    assert all(value.startswith('data:image/jpeg;base64,') for value in pictures.values())
    assert stats['chlorosis_pct'] == 0
    assert stats['total_pixels'] > 0


@pytest.mark.parametrize('color', [(0, 150, 0), (0, 220, 220), (0, 0, 0)])
def test_empty_overlay_regions(color):
    image = np.full((16, 16, 3), color, dtype=np.uint8)
    mask = np.full((16, 16), 255, dtype=np.uint8)
    pictures, stats = generate_visual_pipeline(image, mask, None)
    assert all(pictures.values())
    assert sum(stats[k] for k in ['healthy_pct', 'chlorosis_pct', 'necrosis_pct']) == 100


@pytest.mark.parametrize('name', ['healthy_006.JPG', 'healthy_008.JPG'])
def test_healthy_prediction_regression(client, name):
    response = upload(client, (HEALTHY / name).read_bytes())
    assert response.status_code == 200
    assert response.get_json()['prediction'] in web.classes


def test_no_file(client):
    assert_error(client.post('/predict'), 400, 'missing_image')


def test_empty_filename(client):
    assert_error(upload(client, b'', ''), 400, 'missing_filename')


def test_bad_file(client):
    assert_error(upload(client, b'not an image'), 400, 'invalid_image')


def test_disallowed_extension(client):
    assert_error(upload(client, name='payload.html'), 415, 'unsupported_extension')


def test_disallowed_format(client):
    from PIL import Image
    blob = io.BytesIO()
    Image.new('RGB', (10, 10)).save(blob, format='GIF')
    assert_error(upload(client, blob.getvalue(), 'fake.png'), 415, 'unsupported_image')


def test_upload_byte_limit(client, monkeypatch):
    monkeypatch.setitem(web.app.config, 'MAX_CONTENT_LENGTH', 100)
    assert_error(upload(client), 413, 'http_413')


def test_pixel_limit_before_decode(client, monkeypatch):
    monkeypatch.setitem(web.app.config, 'MAX_IMAGE_PIXELS', 100)
    def forbidden(*args, **kwargs):
        pytest.fail('OpenCV must not decode an oversized image')
    monkeypatch.setattr(web.cv2, 'imread', forbidden)
    assert_error(upload(client), 413, 'image_too_large')


def test_no_leaf(client):
    image = cv2.imencode('.png', np.full((40, 40, 3), 255, np.uint8))[1].tobytes()
    assert_error(upload(client, image, 'blank.png'), 422, 'no_leaf')


@pytest.mark.parametrize('failure', [False, True])
def test_unique_safe_path_and_cleanup(client, monkeypatch, failure):
    seen = []
    original = web.analyze_leaf_full
    def track(path):
        path = Path(path)
        seen.append(path)
        assert path.exists()
        assert len(path.stem) == 32 and int(path.stem, 16) >= 0
        assert path.suffix == '.jpg'
        if failure:
            raise RuntimeError('sensitive internal details')
        return original(str(path))
    monkeypatch.setattr(web, 'analyze_leaf_full', track)
    for name in ['../../overwrite.jpg', '/tmp/overwrite.jpg']:
        response = upload(client, name=name)
        if failure:
            assert_error(response, 500, 'processing_error')
            assert 'sensitive internal' not in response.get_data(as_text=True)
        else:
            assert response.status_code == 200
    assert len(set(seen)) == 2
    assert all(not p.exists() and not p.parent.exists() for p in seen)


def test_invalid_upload_cleanup(client, monkeypatch):
    seen = []
    original = web.validate_image
    def track(path):
        seen.append(Path(path))
        return original(path)
    monkeypatch.setattr(web, 'validate_image', track)
    assert_error(upload(client, b'bad'), 400, 'invalid_image')
    assert seen and not seen[0].parent.exists()


@pytest.mark.parametrize('corrupt', [False, True])
def test_model_load_failure(client, tmp_path, caplog, corrupt):
    path = tmp_path / 'model.pkl'
    if corrupt:
        path.write_bytes(b'not a joblib model')
    assert not web.load_model_bundle(path)
    assert 'Startup model error' in caplog.text
    assert web.primary_model is None
    assert_error(upload(client), 503, 'model_unavailable')


def test_schema_failure(client, monkeypatch):
    monkeypatch.setattr(web, 'feature_names', ['missing_feature'])
    assert_error(upload(client), 503, 'schema_error')


def test_model_failure(client, monkeypatch):
    class Broken:
        def predict(self, X):
            raise RuntimeError('private model error')
    monkeypatch.setattr(web, 'primary_model', Broken())
    assert_error(upload(client), 503, 'model_error')


def test_probability_schema_failure(client, monkeypatch):
    class Broken:
        classes_ = web.classes
        def predict(self, X):
            return [web.classes[0]]
        def predict_proba(self, X):
            return [[1, 0]]
    monkeypatch.setattr(web, 'primary_model', Broken())
    assert_error(upload(client), 503, 'model_error')


def test_unexpected_exception(client, monkeypatch):
    def fail(*args):
        raise RuntimeError('private internals')
    monkeypatch.setattr(web, 'compute_visual_stress_indicator', fail)
    assert_error(upload(client), 500, 'internal_error')


def test_http_errors_json(client):
    assert_error(client.get('/not-found'), 404, 'http_404')
    response = client.get('/predict')
    assert_error(response, 405, 'http_405')
    assert 'POST' in response.headers['Allow']


def test_cwd_independent_model_loading(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-c', 'import app, config; assert app.primary_model is not None; assert config.CSV_PATH.is_file(); assert app.benchmark_data'], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_startup_error(tmp_path):
    env = dict(os.environ, LECOR_MODEL_PATH=str(tmp_path / 'missing.pkl'), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, str(ROOT / 'app.py')], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Startup error: Cannot load model' in result.stderr


def test_safe_server_defaults(monkeypatch):
    for name in ['LECOR_HOST', 'LECOR_DEBUG', 'PORT']:
        monkeypatch.delenv(name, raising=False)
    assert config.server_options() == {'host': '127.0.0.1', 'debug': False, 'port': 5001}
    monkeypatch.setenv('LECOR_HOST', '0.0.0.0')
    monkeypatch.setenv('LECOR_DEBUG', 'true')
    assert config.server_options()['host'] == '0.0.0.0'
    assert config.server_options()['debug'] is True


def test_versioned_paths():
    a = config.versioned_path('leaf_features', '.csv')
    b = config.versioned_path('leaf_features', '.csv')
    assert a != b and a.parent == ROOT and '_v1_' in a.name
    assert a != config.CSV_PATH


def test_originals_and_backups_unchanged():
    manifest = json.loads((ROOT / 'backup_original' / 'sha256_manifest.json').read_text())
    for name, expected in manifest.items():
        for base in [ROOT, ROOT / 'backup_original']:
            assert hashlib.sha256((base / name).read_bytes()).hexdigest() == expected, name


def test_regeneration_uses_new_artifact_names(tmp_path, monkeypatch):
    import runpy
    monkeypatch.setattr(config, 'BASE_DIR', tmp_path)
    runpy.run_path(str(ROOT / 'build_dataset.py'), run_name='__main__')
    csv_files = list(tmp_path.glob('leaf_features_v1_*.csv'))
    assert len(csv_files) == 1
    monkeypatch.setattr(config, 'CSV_PATH', csv_files[0])
    # New gate leaves no Moderate/Severe examples: do not fit a misleading two-class model.
    with pytest.raises(ValueError, match='lacks one or more required classes'):
        runpy.run_path(str(ROOT / 'train_model.py'), run_name='__main__')
    assert csv_files[0].with_suffix('.metadata.json').exists()
    assert not list(tmp_path.glob('leaf_stress_model_v1_*.pkl'))
    assert not list(tmp_path.glob('model_benchmark_v1_*.json'))
    assert not (tmp_path / 'leaf_features.csv').exists()
    assert not (tmp_path / 'leaf_stress_model.pkl').exists()
    assert not (tmp_path / 'model_benchmark.json').exists()
