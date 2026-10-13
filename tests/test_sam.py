"""SAM isolation and API errors, without requiring model downloads in unit tests."""
import base64
import io
import json

import cv2
import numpy as np
import pytest

import sam_segmentation as sam


@pytest.mark.parametrize('points', [None, [], [{'x': .2, 'y': .3, 'label': 0}],
    [{'x': float('nan'), 'y': .2, 'label': 1}], [{'x': 1.1, 'y': 0, 'label': 1}],
    [{'x': True, 'y': 0, 'label': 1}], [{'x': 0, 'y': 0, 'label': '1'}],
    [{'x': 0, 'y': 0, 'label': 1, 'extra': 0}], [{'x': 0, 'y': 0, 'label': 1}]*65])
def test_invalid_prompts(points):
    with pytest.raises(sam.SAMError) as exc:
        sam.validate_points(points)
    assert exc.value.code == 'invalid_points' and exc.value.status == 400


class Predictor:
    def __init__(self):
        self.reset = False

    def set_image(self, image):
        self.shape = image.shape[:2]

    def predict(self, **kwargs):
        self.prompts = kwargs
        mask = np.zeros(self.shape, bool)
        mask[20:70, 20:70] = True
        return mask[None], np.array([.85]), None

    def reset_image(self):
        self.reset = True


def picture(tmp_path):
    image = np.full((100, 150, 3), (20, 150, 45), np.uint8)
    path = tmp_path/'green.png'
    assert cv2.imwrite(str(path), image)
    return path


def decode(url):
    return cv2.imdecode(np.frombuffer(base64.b64decode(url.split(',')[1]), np.uint8), cv2.IMREAD_UNCHANGED)


def test_background_removed_and_prompts_scaled(tmp_path, monkeypatch):
    predictor = Predictor(); monkeypatch.setattr(sam, '_predictor', predictor)
    result = sam.segment(picture(tmp_path), [{'x': .3, 'y': .4, 'label': 1}, {'x': .9, 'y': .9, 'label': 0}])
    image, mask = decode(result['image']), decode(result['mask'])
    assert np.all(image[mask == 0] == 220)
    assert np.all(image[mask > 0] == (20, 150, 45))
    assert predictor.reset and not sam._lock.locked()
    assert np.allclose(predictor.prompts['point_coords'], [[44.7, 39.6], [134.1, 89.1]])
    # The unchanged whole-plant colour extractor now sees target foliage only.
    import plant_features as pf
    isolated = tmp_path/'isolated.png'; cv2.imwrite(str(isolated), image)
    _, foliage, _, quality = pf.prepare_plant(isolated)
    assert quality['passed'] and not quality['border_contact']
    assert not foliage[90, 130] and foliage[40, 40]


def test_prompt_mismatch_and_lock_release(tmp_path, monkeypatch):
    monkeypatch.setattr(sam, '_predictor', Predictor())
    with pytest.raises(sam.SAMError, match='does not match'):
        sam.segment(picture(tmp_path), [{'x': .9, 'y': .9, 'label': 1}])
    assert not sam._lock.locked()


def test_missing_checkpoint_does_not_load(monkeypatch, tmp_path):
    monkeypatch.setattr(sam, '_predictor', None)
    monkeypatch.setattr(sam, 'CHECKPOINT', tmp_path/'missing.pth')
    assert not sam.status()['available']
    with pytest.raises(sam.SAMError, match='setup required'):
        sam._get_predictor()


def test_sam_api_validates_upload_and_returns_reviewable_image(tmp_path, monkeypatch):
    import app as web
    monkeypatch.setattr(web.config, 'ANALYSIS_MODE', 'wholeplant')
    monkeypatch.setattr(sam, '_predictor', Predictor())
    client = web.app.test_client()
    points = json.dumps([{'x': .3, 'y': .4, 'label': 1}, {'x': .9, 'y': .9, 'label': 0}])
    assert client.post('/api/sam/segment', data={'points':'bad'}).status_code == 400
    assert client.post('/api/sam/segment', data={'points':points}).status_code == 400
    response = client.post('/api/sam/segment', data={'points':points, 'image':(io.BytesIO(b'bad'), 'x.png')})
    assert response.status_code == 400 and response.json['code'] == 'invalid_image'
    response = client.post('/api/sam/segment', data={'points':points, 'image':(io.BytesIO(picture(tmp_path).read_bytes()), '../../x.png')})
    assert response.status_code == 200 and response.json['model'] == 'SAM ViT-B'
    assert response.json['image'].startswith('data:image/png;base64,')
    # Feed the exact exported preparation through the existing analysis endpoint.
    monkeypatch.setattr(web, 'analyze_leaf_full', __import__('plant_features').analyze_plant_full)
    prepared = base64.b64decode(response.json['image'].split(',')[1])
    response = client.post('/predict', data={'mask_preview':'true', 'input_preparation':'sam',
        'image':(io.BytesIO(prepared),'target-plant-sam.png')})
    assert response.status_code == 200 and response.json['code'] == 'mask_review'
    assert response.json['quality']['analysis_scope'].endswith('SAM region')
    assert any('SAM-assisted' in warning for warning in response.json['quality']['warnings'])
    monkeypatch.setattr(web.config, 'ANALYSIS_MODE', 'leaf')
    assert client.get('/api/sam/status').status_code == 404
