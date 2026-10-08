"""Shared RGB/HSV feature extraction and 2D projected shape descriptors.

Also produces color-threshold visualizations and an experimental heuristic
indicator. None of these outputs measures physiological water status.
"""

import base64
import cv2
import numpy as np


def segment_leaf(img_bgr):
    """Isolate the leaf from the background. Returns (mask, largest_contour)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Range covering green through yellow/chlorotic and dry brown leaves
    lower = np.array([10, 20, 20])
    upper = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, None

    largest = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)
    return clean_mask, largest


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
    """Returns a dict of features for one image, or None if no leaf found."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    mask, contour = segment_leaf(img)
    if contour is None:
        return None

    color_feats = extract_color_features(img, mask)
    if color_feats is None:
        return None
    shape_feats = extract_shape_features(contour)

    return {**color_feats, **shape_feats}


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
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
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
    """
    Comprehensive pipeline returning extracted features, intermediate CV stages,
    and heuristic color-group pixel fractions.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None

    mask, contour = segment_leaf(img)
    if contour is None:
        return None, None, None

    color_feats = extract_color_features(img, mask)
    if color_feats is None:
        return None, None, None
    shape_feats = extract_shape_features(contour)
    features = {**color_feats, **shape_feats}

    visualizations, tissue_stats = generate_visual_pipeline(img, mask, contour)
    return features, visualizations, tissue_stats
