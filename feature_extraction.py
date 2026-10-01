"""
feature_extraction.py
----------------------
Turns a leaf photo into a small set of numeric features:
  - Color features (HSV stats + vegetation indices) -> main drought signal
  - Shape/curl features (solidity, aspect ratio, compactness) -> proxy for
    "leaf angle" since single cropped leaf photos can't show true
    stem-to-leaf insertion angle, but DO show curling/rolling, which is a
    real drought symptom.
"""

import cv2
import numpy as np


def segment_leaf(img_bgr):
    """Isolate the leaf from the background. Returns (mask, largest_contour)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Loose range: covers green through yellow/brown, since stressed leaves
    # aren't pure green. Tune these if your backgrounds are also green.
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
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0
    extent = area / (w * h) if w * h > 0 else 0

    perimeter = cv2.arcLength(contour, True)
    # Compactness: 1.0 = perfect circle, drops as the outline gets more
    # irregular/curled. Used here as the "curl index".
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
