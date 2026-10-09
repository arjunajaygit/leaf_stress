"""Shared RGB/HSV feature extraction and 2D projected shape descriptors.

Also produces color-threshold visualizations and an experimental heuristic
indicator. None of these outputs measures physiological water status.
"""

import base64
import config
import cv2
import numpy as np


QUALITY_MESSAGES = {
    "unreadable_image": "The file could not be read as an image. Retake photo or choose another file.",
    "no_leaf": "No foreground leaf candidate was found. Retake photo with one leaf against a contrasting background.",
    "border_contact": "The selected mask touches the frame; the leaf may be cut off or merged with background. Retake photo with the full leaf inside the frame.",
    "foreground_too_large": "The selected mask fills too much of the frame. Retake photo with a visible neutral background around the leaf.",
    "foreground_too_small": "The selected mask is too small in the frame. Retake photo closer to one fully visible leaf.",
    "multiple_components": "Several large foreground regions were found. Retake photo showing only one leaf.",
    "tiny_contour": "The detected outline is too small for reliable shape measurements. Retake photo with a larger, clearer leaf.",
}


def standardize_image(image):
    """Preserve aspect ratio and set the longest side to 800 px by default.

    Upsampling small inputs standardizes geometry but adds no image information.
    """
    height, width = image.shape[:2]
    scale = config.WORKING_LONGEST_SIDE / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if size == (width, height):
        return image.copy()
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def morphology_kernel_size(shape):
    size = max(1, round(config.MORPH_KERNEL_AT_REFERENCE * max(shape[:2]) / config.MORPH_REFERENCE_SIDE))
    return size if size % 2 else size + 1


def _segment_candidates(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    candidates = cv2.inRange(hsv, np.array(config.HSV_LOWER), np.array(config.HSV_UPPER))
    size = morphology_kernel_size(image.shape)
    kernel = np.ones((size, size), np.uint8)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, kernel)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea) if contours else None
    selected = np.zeros_like(candidates)
    if largest is not None:
        cv2.drawContours(selected, [largest], -1, 255, -1)
    return selected, largest, candidates


def segment_leaf(img_bgr):
    """Segment a prepared image; shared callers standardize before this step."""
    mask, contour, _ = _segment_candidates(img_bgr)
    return mask, contour


def border_contact_fraction(contour, shape):
    """Euclidean contour length on frame edges / total external contour length.

    A vertex-only contact contributes zero length. No contour or zero perimeter
    returns 0. This diagnostic does not change the Boolean border rejection rule.
    """
    if contour is None:
        return 0.0
    points = contour.reshape(-1, 2).astype(float)
    following = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(following - points, axis=1)
    perimeter = float(lengths.sum())
    if perimeter == 0:
        return 0.0
    height, width = shape[:2]
    on_frame = (((points[:, 0] == 0) & (following[:, 0] == 0))
                | ((points[:, 0] == width - 1) & (following[:, 0] == width - 1))
                | ((points[:, 1] == 0) & (following[:, 1] == 0))
                | ((points[:, 1] == height - 1) & (following[:, 1] == height - 1)))
    return float(lengths[on_frame].sum() / perimeter)


def check_mask_quality(mask, contour, candidates):
    """Retain all failures, inspecting components before selecting/filling one outline."""
    pixels = mask.size
    count, _, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([])
    cutoff = max(pixels * config.MASK_LARGE_COMPONENT_MIN_FRACTION,
                 float(areas.max()) * config.MASK_LARGE_COMPONENT_RELATIVE_FRACTION if len(areas) else 0)
    large = int(np.sum(areas >= cutoff)) if len(areas) else 0
    fraction = float(np.count_nonzero(mask) / pixels)
    border = bool(np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1]))
    area = float(cv2.contourArea(contour)) if contour is not None else 0.0
    reasons = []
    if contour is None:
        reasons.append("no_leaf")
    else:
        if border and config.MASK_REJECT_BORDER_CONTACT:
            reasons.append("border_contact")
        if fraction > config.MASK_MAX_FOREGROUND_FRACTION:
            reasons.append("foreground_too_large")
        if fraction < config.MASK_MIN_FOREGROUND_FRACTION:
            reasons.append("foreground_too_small")
        if large > config.MASK_MAX_LARGE_COMPONENTS:
            reasons.append("multiple_components")
        if area < max(config.MASK_MIN_CONTOUR_AREA_PIXELS, pixels * config.MASK_MIN_CONTOUR_AREA_FRACTION):
            reasons.append("tiny_contour")
    return {
        "passed": not reasons, "reason_code": reasons[0] if reasons else "ok",
        "reason_codes": reasons,
        "message": "Mask passed provisional quality checks; leaf identity is not verified." if not reasons else " ".join(QUALITY_MESSAGES[r] for r in reasons),
        "border_contact": border, "border_contact_fraction": border_contact_fraction(contour, mask.shape),
        "foreground_fraction": fraction,
        "candidate_foreground_fraction": float(np.count_nonzero(candidates) / pixels),
        "components": int(count - 1), "large_components": large,
        "contour_area": area, "working_width": int(mask.shape[1]), "working_height": int(mask.shape[0]),
    }


def prepare_leaf(image_path):
    """One preprocessing, gate and feature path used by training, inference and reports."""
    original = cv2.imread(str(image_path))
    if original is None:
        return None, None, None, None, {
            "passed": False, "reason_code": "unreadable_image", "reason_codes": ["unreadable_image"],
            "message": QUALITY_MESSAGES["unreadable_image"],
        }
    image = standardize_image(original)
    mask, contour, candidates = _segment_candidates(image)
    quality = check_mask_quality(mask, contour, candidates)
    features = None
    if quality["passed"]:
        features = {**extract_color_features(image, mask), **extract_shape_features(contour)}
    return image, mask, contour, features, quality


def extract_color_features(img_bgr, mask):
    """Extract statistical and vegetation index features from leaf pixels."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv_pixels = hsv[mask == 255]
    bgr_pixels = img_bgr[mask == 255].astype(float)

    if len(hsv_pixels) == 0:
        return None

    h, s, v = hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2]
    b, g, r = bgr_pixels[:, 0], bgr_pixels[:, 1], bgr_pixels[:, 2]

    # RGB greenness indices; not direct measurements of water or chlorophyll content
    exg = 2 * g - r - b
    gli = (2 * g - r - b) / (2 * g + r + b + 1e-6)

    return {
        "h_mean": float(h.mean()), "h_std": float(h.std()),
        "s_mean": float(s.mean()), "s_std": float(s.std()),
        "v_mean": float(v.mean()), "v_std": float(v.std()),
        "r_mean": float(r.mean()), "g_mean": float(g.mean()), "b_mean": float(b.mean()),
        "exg_mean": float(exg.mean()),
        "gli_mean": float(gli.mean()),
    }


def extract_shape_features(contour):
    """Extract 2D projected shape descriptors; these do not directly measure curl."""
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0
    extent = area / (w * h) if w * h > 0 else 0

    perimeter = cv2.arcLength(contour, True)
    compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    return {
        "solidity": float(solidity),
        "aspect_ratio": float(aspect_ratio),
        "extent": float(extent),
        "compactness": float(compactness),
    }


def extract_all_features(image_path):
    """Return shared standardized features only for images passing the quality gate."""
    return prepare_leaf(image_path)[3]


def _img_to_base64(img_bgr):
    """Helper to convert BGR image to base64 JPEG string."""
    success, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not success:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


def generate_visual_pipeline(img_bgr, mask, contour):
    """
    Generates computer vision pipeline stages for visual inspection:
    1. Segmented leaf cutout with contour border
    2. Color-threshold map (yellow-toned, dark/brown-toned and remaining pixels)
    3. GLI (Green Leaf Index) RGB greenness heatmap
    """
    h, w = img_bgr.shape[:2]
    # Resize large images for fast web response while preserving aspect ratio
    max_dim = 640
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea) if contours else None

    # 1. Clean Segmented Cutout with highlighted neon border
    segmented = np.zeros_like(img_bgr)
    segmented[mask == 255] = img_bgr[mask == 255]
    if contour is not None:
        cv2.drawContours(segmented, [contour], -1, (0, 255, 128), 2)

    # 2. Heuristic color grouping; these are not tissue diagnoses.
    # Legacy variable names below refer only to color groups.
    # Yellow-toned: H in [18, 35], S >= 45, excluding dark/brown group.
    # Dark/brown-toned: H < 18 or (V < 75 and S > 35).
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    leaf_indices = np.where(mask == 255)
    total_leaf_pixels = len(leaf_indices[0])

    pathology_vis = segmented.copy()
    chlorosis_pixels = 0
    necrosis_pixels = 0
    healthy_pixels = 0

    if total_leaf_pixels > 0:
        h_chan = hsv[:, :, 0]
        s_chan = hsv[:, :, 1]
        v_chan = hsv[:, :, 2]

        # Dark/brown group takes precedence; remaining pixels are not necessarily healthy.
        necrosis_mask = (mask == 255) & ((h_chan < 18) | ((v_chan < 75) & (s_chan > 35)))
        chlorosis_mask = (mask == 255) & (~necrosis_mask) & (h_chan >= 18) & (h_chan <= 35) & (s_chan >= 45)
        healthy_mask = (mask == 255) & (~chlorosis_mask) & (~necrosis_mask)

        chlorosis_pixels = int(np.sum(chlorosis_mask))
        necrosis_pixels = int(np.sum(necrosis_mask))
        healthy_pixels = int(np.sum(healthy_mask))

        # Color overlay for threshold groups
        # Yellow-toned -> Bright Gold/Yellow highlight (0, 220, 255) BGR
        # Dark/brown-toned -> Crimson/Red highlight (0, 0, 240) BGR
        # Remaining pixels retain their original color.
        if chlorosis_pixels:
            pathology_vis[chlorosis_mask] = cv2.addWeighted(
                pathology_vis[chlorosis_mask], 0.35,
                np.full_like(pathology_vis[chlorosis_mask], (0, 220, 255)), 0.65, 0
            )
        if necrosis_pixels:
            pathology_vis[necrosis_mask] = cv2.addWeighted(
                pathology_vis[necrosis_mask], 0.3,
                np.full_like(pathology_vis[necrosis_mask], (40, 40, 255)), 0.7, 0
            )

    # 3. RGB greenness heatmap (Green Leaf Index - GLI)
    bgr_f = img_bgr.astype(np.float32)
    b, g, r = bgr_f[:, :, 0], bgr_f[:, :, 1], bgr_f[:, :, 2]
    gli_map = (2.0 * g - r - b) / (2.0 * g + r + b + 1e-6)

    # Clip GLI to [-0.15, 0.35] and normalize to 0..255 for display only
    gli_clipped = np.clip(gli_map, -0.15, 0.35)
    gli_norm = ((gli_clipped - (-0.15)) / (0.35 - (-0.15)) * 255.0).astype(np.uint8)
    gli_colored = cv2.applyColorMap(gli_norm, cv2.COLORMAP_TURBO)
    gli_heatmap = np.zeros_like(img_bgr)
    gli_heatmap[mask == 255] = gli_colored[mask == 255]

    tissue_stats = {
        "total_pixels": total_leaf_pixels,
        "healthy_pct": round((healthy_pixels / total_leaf_pixels * 100), 1) if total_leaf_pixels > 0 else 0,
        "chlorosis_pct": round((chlorosis_pixels / total_leaf_pixels * 100), 1) if total_leaf_pixels > 0 else 0,
        "necrosis_pct": round((necrosis_pixels / total_leaf_pixels * 100), 1) if total_leaf_pixels > 0 else 0,
    }

    visualizations = {
        "segmented": _img_to_base64(segmented),
        "pathology": _img_to_base64(pathology_vis),
        "gli_heatmap": _img_to_base64(gli_heatmap),
    }

    return visualizations, tissue_stats


def compute_visual_stress_indicator(features, prediction, probabilities):
    """Return an integer heuristic on 0–100, or None for confounded results.

    Original arbitrary weights are retained for continuity, not validated as a
    physiological scale. Class probabilities are not water-stress severity.
    """
    if prediction == "Disease_Confounded":
        return None
    base = (probabilities.get("Healthy", 0.0) * 10.0
            + probabilities.get("Moderate_Water_Stress", 0.0) * 55.0
            + probabilities.get("Severe_Water_Stress", 0.0) * 92.0
            + probabilities.get("Disease_Confounded", 0.0) * 45.0)
    # Unvalidated shape and greenness adjustments retained from the prototype.
    solidity = features.get("solidity", 0.9)
    compactness = features.get("compactness", 0.5)
    shape_adjustment = max(0.0, (0.95 - solidity) * 30.0) + max(0.0, (0.50 - compactness) * 20.0)
    gli = features.get("gli_mean", 0.15)
    color_adjustment = max(0.0, (0.20 - gli) * 50.0)
    value = min(100.0, max(0.0, base + 0.25 * shape_adjustment + 0.25 * color_adjustment))
    return int(round(value))


def generate_agronomic_advisory(prediction):
    """Suggest independent checks, without diagnosing a cause or prescribing treatment."""
    guidance = {
        "Healthy": "The image resembles the dataset's Healthy category. This does not rule out water stress or other problems.",
        "Moderate_Water_Stress": "The image resembles the dataset's Moderate Water Stress category. Confirm plant and soil conditions directly.",
        "Severe_Water_Stress": "The image resembles the dataset's Severe Water Stress category. Inspect the plant promptly; this image does not establish the cause.",
        "Disease_Confounded": "Disease-like or other confounding symptoms suspected. Water-stress severity is not reported for this result.",
    }
    return {
        "guidance": guidance.get(prediction, "The visual result is unsupported. Inspect the plant directly."),
        "action_items": [
            "Inspect the whole plant and both sides of affected leaves.",
            "Check soil moisture directly before deciding whether watering is needed.",
            "Consider disease, pests, nutrient problems, heat and other possible causes; visual symptoms can overlap.",
            "Consult a qualified agricultural expert if symptoms persist, worsen or remain unclear. Do not choose a treatment from this image alone.",
        ],
    }


def analyze_leaf_full(image_path):
    """Return features, displays, color fractions and quality for every readable input."""
    image, mask, contour, features, quality = prepare_leaf(image_path)
    if image is None:
        return None, {}, {}, quality
    # Binary previews show both the selected mask and all pre-selection candidates.
    _, _, candidates = _segment_candidates(image)
    previews = {}
    for name, binary in (("mask", mask), ("candidate_mask", candidates)):
        scale = min(1.0, 640 / max(binary.shape))
        display = cv2.resize(binary, (max(1, round(binary.shape[1] * scale)),
                                      max(1, round(binary.shape[0] * scale))), interpolation=cv2.INTER_NEAREST)
        previews[name] = _img_to_base64(display)
    if not quality["passed"]:
        return None, previews, {}, quality
    visualizations, tissue_stats = generate_visual_pipeline(image, mask, contour)
    return features, {**visualizations, **previews}, tissue_stats, quality


def extract_experimental_features(img_bgr, mask, contour):
    """Ablation-only descriptors; never added to production extract_all_features."""
    pixels = img_bgr[mask == 255].astype(float)
    if not len(pixels) or contour is None:
        raise ValueError('Experimental features require a nonempty mask and contour')
    sums = pixels.sum(axis=1)
    chromaticity = np.divide(pixels, sums[:, None], out=np.zeros_like(pixels), where=sums[:, None] > 0)
    # Float OpenCV conversion yields physical CIELAB a*/b*, not uint8 offsets.
    lab = cv2.cvtColor(img_bgr.astype(np.float32) / 255., cv2.COLOR_BGR2LAB)[mask == 255]
    width, height = cv2.minAreaRect(contour)[1]
    return {'chrom_r_mean': float(chromaticity[:, 2].mean()),
            'chrom_g_mean': float(chromaticity[:, 1].mean()),
            'lab_a_mean': float(lab[:, 1].mean()), 'lab_a_std': float(lab[:, 1].std()),
            'lab_b_mean': float(lab[:, 2].mean()), 'lab_b_std': float(lab[:, 2].std()),
            'oriented_axis_ratio': float(min(width, height) / max(width, height)) if max(width, height) > 0 else 0.}
