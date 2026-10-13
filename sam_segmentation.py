"""Optional local, point-guided Meta SAM. No classifier or downloads at import time."""
import base64
import importlib.util
import os
from pathlib import Path
import threading

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path(os.environ.get('LECOR_SAM_CHECKPOINT', ROOT / 'models' / 'sam_vit_b_01ec64.pth'))
if not CHECKPOINT.is_absolute():
    CHECKPOINT = ROOT / CHECKPOINT
DEVICE = os.environ.get('LECOR_SAM_DEVICE', 'cpu')
MAX_SIDE = 800
_lock = threading.Lock()
_predictor = None


class SAMError(Exception):
    def __init__(self, message, code='sam_unavailable', status=503):
        super().__init__(message)
        self.code, self.status = code, status


def status():
    missing = [name for name in ('torch', 'torchvision', 'segment_anything') if importlib.util.find_spec(name) is None]
    ready = not missing and CHECKPOINT.is_file()
    return dict(available=ready, model='SAM ViT-B', device=DEVICE,
                message='SAM ready; first use loads the model.' if ready else
                'SAM setup required. Run the installation steps in docs/sam_setup.md.',
                missing_packages=missing, checkpoint_present=CHECKPOINT.is_file())


def validate_points(points):
    if not isinstance(points, list) or not 1 <= len(points) <= 64:
        raise SAMError('Add 1–64 keep/exclude points, including at least one keep point.', 'invalid_points', 400)
    result = []
    for point in points:
        if not isinstance(point, dict) or set(point) != {'x', 'y', 'label'}:
            raise SAMError('Each point needs x, y and label.', 'invalid_points', 400)
        x, y, label = point['x'], point['y'], point['label']
        if (type(x) not in (int, float) or type(y) not in (int, float) or
                not np.isfinite([x, y]).all() or not 0 <= x <= 1 or not 0 <= y <= 1 or
                type(label) is not int or label not in (0, 1)):
            raise SAMError('Point coordinates must be between 0 and 1; label must be 0 or 1.', 'invalid_points', 400)
        result.append((x, y, label))
    if not any(p[2] == 1 for p in result):
        raise SAMError('Mark at least one part of the target plant with Keep.', 'invalid_points', 400)
    return result


def _get_predictor():
    global _predictor
    if _predictor is None:
        if not status()['available']:
            raise SAMError(status()['message'])
        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            # A local trusted checkpoint from Meta is required; never a user upload.
            model = sam_model_registry['vit_b'](checkpoint=str(CHECKPOINT))
            model.to(device=DEVICE)
            model.eval()
            _predictor = SamPredictor(model)
        except Exception as exc:
            raise SAMError('SAM could not load. Check its checkpoint, PyTorch installation and LECOR_SAM_DEVICE.') from exc
    return _predictor


def png_url(image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise SAMError('Could not encode the segmented image.', 'sam_encoding_error', 500)
    return 'data:image/png;base64,' + base64.b64encode(encoded).decode('ascii')


def segment(path, points):
    points = validate_points(points)
    image = cv2.imread(str(path))
    if image is None:
        raise SAMError('Image cannot be decoded.', 'invalid_image', 400)
    height, width = image.shape[:2]
    scale = min(1., MAX_SIDE / max(height, width))
    if scale < 1:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    height, width = image.shape[:2]
    coordinates = np.array([(x * (width-1), y * (height-1)) for x, y, _ in points], dtype=np.float32)
    labels = np.array([p[2] for p in points], dtype=np.int32)
    if not _lock.acquire(blocking=False):
        raise SAMError('SAM is processing another image. Try again shortly.', 'sam_busy', 409)
    try:
        predictor = _get_predictor()
        try:
            predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            masks, scores, _ = predictor.predict(point_coords=coordinates, point_labels=labels,
                                                multimask_output=False)
        finally:
            predictor.reset_image()
        mask = np.asarray(masks[0], dtype=bool)
        if mask.shape != (height, width) or not np.isfinite(scores).all():
            raise SAMError('SAM returned an invalid mask.', 'sam_invalid_mask', 500)
        if np.count_nonzero(mask) < 128:
            raise SAMError('Selected area is too small. Add Keep points on more of the plant.', 'sam_empty_mask', 422)
        # Do not quietly accept a mask contradicting the user's point prompts.
        indices = np.rint(coordinates).astype(int)
        if any(bool(mask[y, x]) != bool(label) for (x, y), label in zip(indices, labels)):
            raise SAMError('The mask does not match all your marks. Move conflicting points or mark more leaves/background.', 'sam_prompt_mismatch', 422)
        isolated = image.copy()
        isolated[~mask] = 220  # Neutral grey is excluded by the existing foliage extractor.
        overlay = image.copy()
        overlay[mask] = (image[mask] * .55 + np.array([30, 210, 30]) * .45).astype(np.uint8)
        return dict(image=png_url(isolated), mask=png_url(np.uint8(mask)*255), overlay=png_url(overlay),
                    width=width, height=height, selected_fraction=float(mask.mean()),
                    model='SAM ViT-B', message='Review the green overlay. SAM follows your marks; it does not verify tomato identity. The classifier will assess colour-selected foliage inside this region.')
    except SAMError:
        raise
    except Exception as exc:
        raise SAMError('SAM segmentation failed. Check server logs and try a smaller photo.', 'sam_failed', 500) from exc
    finally:
        _lock.release()
