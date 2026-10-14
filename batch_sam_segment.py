"""Batch SAM segmentation for dataset images using automated foliage-guided multi-point prompts.

Loads Meta SAM ViT-B once into memory, generates foliage keep points and background exclude points,
and writes the isolated plant images (background replaced with neutral grey RGB 220) to update the dataset.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

import plant_features as pf

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "dataset_wholeplant_v1_20261013"
DEFAULT_CHECKPOINT = ROOT / "models" / "sam_vit_b_01ec64.pth"
CLASSES = ["Healthy", "Moderate_Water_Stress", "Severe_Water_Stress", "Disease_Confounded"]


def extract_prompts(image):
    """Generate high-confidence Keep points on the plant foliage, an Exclude point, and bounding box."""
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    candidates = cv2.inRange(hsv, np.array(pf.SETTINGS["hsv_lower"]), np.array(pf.SETTINGS["hsv_upper"]))
    b, g, r = cv2.split(image.astype(float))
    candidates[(g < pf.SETTINGS["green_red_ratio_min"] * r) |
               (g < pf.SETTINGS["green_blue_ratio_min"] * b)] = 0

    cand_pixels = int(np.count_nonzero(candidates))
    box = None
    if cand_pixels > 0:
        ys, xs = np.where(candidates > 0)
        box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

    # Extract connected components to cover distinct plant leaves/branches
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])
    sorted_indices = np.argsort(areas)[::-1] if len(areas) else []

    keep_coords = []
    for idx in sorted_indices[:5]:
        if areas[idx] >= 16:
            c_x, c_y = centroids[idx + 1]
            keep_coords.append([float(c_x), float(c_y)])

    if not keep_coords:
        dist_fg = cv2.distanceTransform(candidates, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_fg)
        if max_val > 0:
            keep_coords.append([float(max_loc[0]), float(max_loc[1])])
        else:
            keep_coords.append([float(w / 2), float(h / 2)])

    # Exclude point only if background area is significant (> 15% of frame)
    exclude_coords = []
    if cand_pixels < 0.85 * (w * h):
        dist_bg = cv2.distanceTransform(255 - candidates, cv2.DIST_L2, 5)
        _, max_bg_val, _, max_bg_loc = cv2.minMaxLoc(dist_bg)
        if max_bg_val > 5:
            exclude_coords.append([float(max_bg_loc[0]), float(max_bg_loc[1])])

    coords = np.array(keep_coords + exclude_coords, dtype=np.float32)
    labels = np.array([1] * len(keep_coords) + [0] * len(exclude_coords), dtype=np.int32)
    return coords, labels, box, cand_pixels


def process_dataset(dataset_dir, checkpoint_path, device="cpu", max_side=800):
    dataset_dir = Path(dataset_dir)
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM checkpoint not found at {checkpoint_path}")

    print(f"Loading SAM ViT-B from {checkpoint_path} onto {device}...")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))
    model.to(device=device)
    model.eval()
    predictor = SamPredictor(model)
    print("SAM loaded successfully. Starting batch segmentation...\n")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "checkpoint": str(checkpoint_path),
        "total_images": 0,
        "processed_images": 0,
        "classes": {},
        "records": []
    }

    start_total_time = time.perf_counter()

    for class_name in CLASSES:
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            print(f"Directory {class_dir} not found; skipping.")
            continue

        image_paths = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
        print(f"=== Processing class: {class_name} ({len(image_paths)} images) ===")
        class_stats = {"count": len(image_paths), "avg_mask_fraction": 0.0, "avg_time_sec": 0.0}
        total_class_mask_fraction = 0.0
        total_class_time = 0.0

        for i, img_path in enumerate(image_paths, 1):
            t0 = time.perf_counter()
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  [{i}/{len(image_paths)}] WARNING: Cannot decode {img_path.name}")
                continue

            orig_h, orig_w = image.shape[:2]
            scale = min(1.0, max_side / max(orig_h, orig_w))
            if scale < 1.0:
                resized = cv2.resize(image, (round(orig_w * scale), round(orig_h * scale)), interpolation=cv2.INTER_AREA)
            else:
                resized = image.copy()
            h, w = resized.shape[:2]

            coords, point_labels, box, cand_pixels = extract_prompts(resized)

            predictor.set_image(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            masks, scores, _ = predictor.predict(
                point_coords=coords,
                point_labels=point_labels,
                multimask_output=False
            )

            mask = masks[0]
            mask_pixels = int(np.count_nonzero(mask))

            # If point prompt resulted in too few pixels, retry with bounding box prompt
            if mask_pixels < max(200, int(0.25 * cand_pixels)) and box is not None:
                masks_box, scores_box, _ = predictor.predict(
                    point_coords=coords,
                    point_labels=point_labels,
                    box=box[None, :],
                    multimask_output=False
                )
                if np.count_nonzero(masks_box[0]) > mask_pixels:
                    mask = masks_box[0]
                    scores = scores_box
                    mask_pixels = int(np.count_nonzero(mask))

            predictor.reset_image()
            selected_fraction = float(mask.mean())

            # Isolate plant region; set excluded background to neutral grey 220
            isolated = resized.copy()
            isolated[~mask] = 220

            # Overwrite image in dataset with high-quality JPEG/PNG
            if img_path.suffix.lower() in {".jpg", ".jpeg"}:
                cv2.imwrite(str(img_path), isolated, [cv2.IMWRITE_JPEG_QUALITY, 98])
            else:
                cv2.imwrite(str(img_path), isolated)

            elapsed = time.perf_counter() - t0
            total_class_time += elapsed
            total_class_mask_fraction += selected_fraction

            summary["records"].append({
                "filename": f"{class_name}/{img_path.name}",
                "class": class_name,
                "original_shape": [orig_h, orig_w],
                "processed_shape": [h, w],
                "mask_pixels": mask_pixels,
                "selected_fraction": round(selected_fraction, 4),
                "sam_score": round(float(scores[0]), 4),
                "keep_points": int(np.count_nonzero(point_labels == 1)),
                "time_sec": round(elapsed, 2)
            })

            print(f"  [{i}/{len(image_paths)}] {img_path.name}: {w}x{h}, {mask_pixels} px ({selected_fraction*100:.1f}%), score: {scores[0]:.2f} [{elapsed:.1f}s]")

        if len(image_paths) > 0:
            class_stats["avg_mask_fraction"] = round(total_class_mask_fraction / len(image_paths), 4)
            class_stats["avg_time_sec"] = round(total_class_time / len(image_paths), 2)
        summary["classes"][class_name] = class_stats
        summary["total_images"] += len(image_paths)

    summary["processed_images"] = len(summary["records"])
    summary["total_time_sec"] = round(time.perf_counter() - start_total_time, 2)

    report_path = ROOT / "reports" / f"sam_dataset_update_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2))
    print(f"\nBatch SAM segmentation complete in {summary['total_time_sec']:.1f}s!")
    print(f"Summary report written to {report_path}")
    return summary, report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Target dataset directory to update")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="SAM ViT-B checkpoint path")
    parser.add_argument("--device", type=str, default="cpu", help="Device for SAM (cpu/cuda/mps)")
    parser.add_argument("--max-side", type=int, default=800, help="Maximum image side for processing")
    args = parser.parse_args()

    process_dataset(args.dataset, args.checkpoint, args.device, args.max_side)


if __name__ == "__main__":
    main()
