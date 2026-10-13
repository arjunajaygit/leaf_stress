"""Shared whole-photo foliage features. Masks are candidates, not botanical identification."""
import cv2
import numpy as np
from feature_extraction import extract_color_features, generate_visual_pipeline, _img_to_base64

SCHEMA_VERSION = 'lecor-wholeplant-features-v1'
EXTRACTOR_VERSION = 'foliage-global-regional-v1'
SETTINGS = dict(version='wholeplant-preprocessing-v1', longest_side=800,
                hsv_lower=[5, 25, 20], hsv_upper=[100, 255, 255],
                kernel_at_800=3, min_component_pixels_at_800=16,
                green_red_ratio_min=.90, green_blue_ratio_min=1.05,
                min_foreground_fraction=.005, min_foreground_pixels=128,
                min_original_side=32, min_channel_std=2.0)
COLOR_NAMES = ['h_mean', 'h_std', 's_mean', 's_std', 'v_mean', 'v_std',
               'r_mean', 'g_mean', 'b_mean', 'gli_mean']
FEATURE_NAMES = COLOR_NAMES + ['chrom_r_mean', 'chrom_g_mean', 'lab_a_mean', 'lab_a_std',
    'lab_b_mean', 'lab_b_std', 'yellow_fraction', 'dark_fraction'] + [
    f'{region}_{feature}' for region in ('upper', 'middle', 'lower')
    for feature in ('gli_mean', 'yellow_fraction')]


def preprocessing_signature():
    return {**SETTINGS, 'hsv_lower': list(SETTINGS['hsv_lower']), 'hsv_upper': list(SETTINGS['hsv_upper'])}


def prepare_plant(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError('Image cannot be decoded')
    original_shape = image.shape[:2]
    scale = min(1., SETTINGS['longest_side'] / max(original_shape))
    if scale < 1:
        image = cv2.resize(image, (max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    candidates = cv2.inRange(hsv, np.array(SETTINGS['hsv_lower']), np.array(SETTINGS['hsv_upper']))
    b, g, r = cv2.split(image.astype(float))
    # Exclude many warm soil/skin pixels; still not a semantic plant detector.
    candidates[(g < SETTINGS['green_red_ratio_min'] * r) |
               (g < SETTINGS['green_blue_ratio_min'] * b)] = 0
    k = max(1, round(SETTINGS['kernel_at_800'] * max(image.shape[:2])/800))
    k += 1 - k % 2
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, np.ones((k,k), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates)
    min_area = max(2, round(SETTINGS['min_component_pixels_at_800'] * (max(image.shape[:2])/800)**2))
    keep = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= min_area) + 1
    mask = np.uint8(np.isin(labels, keep)) * 255
    pixels = int(np.count_nonzero(mask)); fraction = pixels/mask.size
    reasons = []
    if min(original_shape) < SETTINGS['min_original_side']: reasons.append('image_too_small')
    if float(image.reshape(-1,3).std(axis=0).max()) < SETTINGS['min_channel_std']: reasons.append('uniform_image')
    if fraction < SETTINGS['min_foreground_fraction'] or pixels < SETTINGS['min_foreground_pixels']: reasons.append('insufficient_foliage')
    border = bool(np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:,0]) or np.any(mask[:,-1]))
    flags = []
    if border: flags.append('Foliage touches the frame: assessment covers visible foliage only; the complete plant is not verified.')
    if fraction > .95: flags.append('Foliage fills the image; background separation and full-plant framing cannot be verified.')
    flags.append('Inspect the mask: green backgrounds, supports, fruit and other plants can be included; brown or shaded foliage can be missed.')
    quality = dict(passed=not reasons, reason_codes=reasons, reason_code=reasons[0] if reasons else 'passed',
                   message='Use a clear plant photo with visible foliage.' if reasons else 'Provisional foliage checks passed.',
                   warnings=flags, border_contact=border, foreground_fraction=fraction,
                   components=int(len(keep)), foreground_pixels=pixels, analysis_scope='visible foliage in this photo')
    return image, mask, candidates, quality


def extract_features(image, mask):
    selected = mask > 0
    if not selected.any(): raise ValueError('Empty foliage mask')
    colors = extract_color_features(image, mask)
    result = {key: colors[key] for key in COLOR_NAMES}
    pixels = image[selected].astype(float)
    chrom = pixels / np.maximum(pixels.sum(axis=1, keepdims=True), 1.)
    lab = cv2.cvtColor(image.astype(np.float32)/255., cv2.COLOR_BGR2LAB)[selected]
    result.update(chrom_r_mean=float(chrom[:,2].mean()), chrom_g_mean=float(chrom[:,1].mean()),
                  lab_a_mean=float(lab[:,1].mean()), lab_a_std=float(lab[:,1].std()),
                  lab_b_mean=float(lab[:,2].mean()), lab_b_std=float(lab[:,2].std()))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dark = (hsv[:,:,0] < 18) | ((hsv[:,:,2] < 75) & (hsv[:,:,1] > 35))
    yellow = (~dark) & (hsv[:,:,0] >= 18) & (hsv[:,:,0] <= 35) & (hsv[:,:,1] >= 45)
    result['yellow_fraction'] = float(yellow[selected].mean())
    result['dark_fraction'] = float(dark[selected].mean())
    b,g,r = cv2.split(image.astype(float)); gli = (2*g-r-b)/(2*g+r+b+1e-6)
    ys = np.where(selected)[0]; boundaries = np.linspace(ys.min(), ys.max()+1, 4).astype(int)
    for index, region in enumerate(('upper','middle','lower')):
        regional = selected.copy(); regional[:boundaries[index]] = False; regional[boundaries[index+1]:] = False
        # Empty bands use global values: no arbitrary zero-valued missing observation.
        result[f'{region}_gli_mean'] = float(gli[regional].mean()) if regional.any() else result['gli_mean']
        result[f'{region}_yellow_fraction'] = float(yellow[regional].mean()) if regional.any() else result['yellow_fraction']
    return {key: float(result[key]) for key in FEATURE_NAMES}


def analyze_plant_full(path):
    image, mask, candidates, quality = prepare_plant(path)
    previews = {'mask': _img_to_base64(mask), 'candidate_mask': _img_to_base64(candidates)}
    if not quality['passed']: return None, previews, {}, quality
    features = extract_features(image, mask)
    visuals, stats = generate_visual_pipeline(image, mask, None)
    return features, {**visuals, **previews}, stats, quality


def relative_indicator(features, prediction, probabilities):
    """Arbitrary ordinal display mapping, neither calibrated severity nor confidence."""
    if prediction == 'Disease_Confounded': return None
    return {'Healthy': 10, 'Moderate_Water_Stress': 50, 'Severe_Water_Stress': 90}[prediction]
