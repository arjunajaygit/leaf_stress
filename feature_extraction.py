"""
feature_extraction.py
----------------------
Extracts biophysical computer vision features from leaf photos:
  - Color features (HSV distributions + vegetation indices EXG, GLI)
  - Morphometric / curl features (solidity, aspect ratio, extent, compactness)
  - Visual pipeline generators (segmented mask, tissue pathology map, GLI heatmap)
  - Agronomic decision support (Crop Water Stress Index, tissue necrosis stats)
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

    # Vegetation indices commonly used in plant-stress literature
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
    """Extract geometric shape and curling descriptors."""
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
    2. Tissue pathology map (Chlorosis / Necrosis / Healthy green decomposition)
    3. GLI (Green Leaf Index) spectral heatmap
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

    # 2. Tissue Pathology Analysis
    # Healthy green: H in [36, 85]
    # Chlorosis (Yellowing): H in [18, 35], S > 50
    # Necrosis (Browning/drying): H < 18 or (V < 70 and S > 30)
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

        # Necrosis takes priority (most severe), then chlorosis, then healthy
        necrosis_mask = (mask == 255) & ((h_chan < 18) | ((v_chan < 75) & (s_chan > 35)))
        chlorosis_mask = (mask == 255) & (~necrosis_mask) & (h_chan >= 18) & (h_chan <= 35) & (s_chan >= 45)
        healthy_mask = (mask == 255) & (~chlorosis_mask) & (~necrosis_mask)

        chlorosis_pixels = int(np.sum(chlorosis_mask))
        necrosis_pixels = int(np.sum(necrosis_mask))
        healthy_pixels = int(np.sum(healthy_mask))

        # Color overlay for pathology map
        # Chlorosis -> Bright Gold/Yellow highlight (0, 220, 255) BGR
        # Necrosis -> Crimson/Red highlight (0, 0, 240) BGR
        # Healthy -> Natural or tinted green (0, 220, 0)
        pathology_vis[chlorosis_mask] = cv2.addWeighted(
            pathology_vis[chlorosis_mask], 0.35,
            np.full_like(pathology_vis[chlorosis_mask], (0, 220, 255)), 0.65, 0
        )
        pathology_vis[necrosis_mask] = cv2.addWeighted(
            pathology_vis[necrosis_mask], 0.3,
            np.full_like(pathology_vis[necrosis_mask], (40, 40, 255)), 0.7, 0
        )

    # 3. Spectral Heatmap (Green Leaf Index - GLI)
    bgr_f = img_bgr.astype(np.float32)
    b, g, r = bgr_f[:, :, 0], bgr_f[:, :, 1], bgr_f[:, :, 2]
    gli_map = (2.0 * g - r - b) / (2.0 * g + r + b + 1e-6)

    # Clip GLI to [-0.2, 0.4] and normalize to 0..255
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


def compute_cwsi(features, prediction, probabilities):
    """
    Computes Crop Water Stress Index (CWSI, 0% to 100%) by fusing model
    prediction probabilities with physical morphology (solidity/extent)
    and vegetation index (GLI).
    """
    p_healthy = probabilities.get("Healthy", 0.0)
    p_moderate = probabilities.get("Moderate_Water_Stress", 0.0)
    p_severe = probabilities.get("Severe_Water_Stress", 0.0)
    p_disease = probabilities.get("Disease_Confounded", 0.0)

    # Base stress from class probabilities
    base_cwsi = (p_moderate * 55.0) + (p_severe * 92.0) + (p_healthy * 10.0) + (p_disease * 45.0)

    # Morphological curl adjustment (wilting leaves lose solidity and compactness)
    solidity = features.get("solidity", 0.9)
    compactness = features.get("compactness", 0.5)
    curl_penalty = max(0.0, (0.95 - solidity) * 30.0) + max(0.0, (0.50 - compactness) * 20.0)

    # Spectral drop adjustment (GLI < 0.15 indicates chlorophyll degradation)
    gli = features.get("gli_mean", 0.15)
    spectral_penalty = max(0.0, (0.20 - gli) * 50.0)

    cwsi = min(100.0, max(0.0, base_cwsi + (0.25 * curl_penalty) + (0.25 * spectral_penalty)))
    return round(float(cwsi), 1)


def generate_agronomic_advisory(prediction, cwsi, tissue_stats):
    """
    Generates tailored agronomic prescriptions for field management.
    """
    chlorosis = tissue_stats.get("chlorosis_pct", 0)
    necrosis = tissue_stats.get("necrosis_pct", 0)

    if prediction == "Healthy":
        status = "Optimal Plant Turgor"
        alert_level = "success"
        irrigation = "Maintain standard baseline irrigation schedule (approx. 1.0 - 1.5 L/m²/day)."
        actions = [
            "Soil moisture levels appear optimal; avoid over-saturation.",
            "Continue periodic leaf visual inspection twice weekly.",
            "Maintain current nutrient fertigation cycle.",
        ]
        recovery_time = "N/A (Healthy canopy)"

    elif prediction == "Moderate_Water_Stress":
        status = "Early / Moderate Abiotic Drought"
        alert_level = "warning"
        irrigation = "Apply 2.5 - 3.5 L/m² via drip irrigation during early morning hours (6:00 AM - 8:00 AM)."
        actions = [
            "Incipient turgor loss detected; hydrate before stomatal closure impacts photosynthesis.",
            "Apply 2-3 inches of organic straw mulch around root zones to suppress soil evaporation.",
            "Verify drip emitter flow rates and inspect for clogged lateral lines.",
            f"Monitor chlorotic margins (currently {chlorosis}% of leaf surface).",
        ]
        recovery_time = "24 to 48 hours following root-zone hydration"

    elif prediction == "Severe_Water_Stress":
        status = "Critical Moisture Deficit / Wilting"
        alert_level = "danger"
        irrigation = "Immediate root-zone pulse watering required (4.0 - 5.0 L/m² split into 2 applications)."
        actions = [
            "Severe turgor depression and cellular flaccidity observed.",
            "Split watering into morning and late afternoon pulses to avoid waterlogging suffocated roots.",
            "Erect temporary shade netting if ambient temperature exceeds 32°C to reduce vapor pressure deficit.",
            f"Necrotic lesion spread detected on {necrosis}% of leaf lamina.",
        ]
        recovery_time = "3 to 5 days with gradual soil re-saturation"

    else:  # Disease_Confounded
        status = "Pathological / Biotic Stress Suspected"
        alert_level = "secondary"
        irrigation = "Hold excessive overhead watering; maintain regulated sub-surface drip only."
        actions = [
            "CAUTION: Foliar symptoms (necrosis/yellowing) resemble drought but stem from biotic pathology.",
            "Refrain from overhead sprinkler irrigation to prevent dispersing bacterial/fungal spores.",
            "Inspect abaxial (underside) leaf surface for fungal hyphae, bacterial lesions, or insect vectors.",
            "Isolate heavily afflicted plants and apply targeted organic copper fungicide or bio-fungicide.",
        ]
        recovery_time = "Dependent on prompt phytosanitary intervention"

    return {
        "status": status,
        "alert_level": alert_level,
        "irrigation_plan": irrigation,
        "action_items": actions,
        "recovery_time": recovery_time,
    }


def analyze_leaf_full(image_path):
    """
    Comprehensive pipeline returning extracted features, intermediate CV stages,
    and biophysical tissue stats.
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
