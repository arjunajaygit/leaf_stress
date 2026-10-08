# Mask quality review — Stage 3

40 byte-unique images: **11 pass, 29 fail**. No images were removed.

Provisional thresholds are acquisition checks, not a validated leaf detector. A pass does not establish leaf identity, label correctness or water status.

Counts below include every reason per image, so they overlap and need not sum to the rejected-image count. Primary counts use the deterministic first reason only.

| Reason | Images failing this reason | Primary reason count |
|---|---:|---:|
| unreadable_image | 0 | 0 |
| no_leaf | 0 | 0 |
| border_contact | 29 | 29 |
| foreground_too_large | 9 | 0 |
| foreground_too_small | 0 | 0 |
| multiple_components | 0 | 0 |
| tiny_contour | 0 | 0 |

| Class | Pass | Fail |
|---|---:|---:|
| Disease_Confounded | 7 | 5 |
| Healthy | 4 | 4 |
| Moderate_Water_Stress | 0 | 9 |
| Severe_Water_Stress | 0 | 11 |

## Threshold snapshot

```json
{
  "preprocessing": {
    "version": "working-resolution-v1",
    "working_longest_side": 800,
    "resize": "area-down-linear-up",
    "morph_reference_side": 800,
    "morph_kernel_at_reference": 5,
    "hsv_lower": [
      10,
      20,
      20
    ],
    "hsv_upper": [
      100,
      255,
      255
    ]
  },
  "quality": {
    "min_foreground_fraction": 0.02,
    "max_foreground_fraction": 0.95,
    "min_contour_area_fraction": 0.005,
    "min_contour_area_pixels": 128,
    "large_component_min_fraction": 0.01,
    "large_component_relative_fraction": 0.2,
    "max_large_components": 1,
    "reject_border_contact": true
  }
}
```

Foreground fraction and border contact refer to the selected filled mask. Components are measured on the full post-morphology candidate mask before choosing the largest contour. A component is large only if its pixel area reaches both the configured image-area fraction and the configured fraction of the largest component.

All images are resized to longest side 800, including upsampling smaller images. The normal morphology kernel is 5×5 at that resolution; smaller direct segmentation inputs receive a scaled odd kernel. Upsampling adds no detail.

## Rejected images

| Filename | All reason codes |
|---|---|
| Disease_Confounded/IMG_4213(1).JPG | border_contact |
| Disease_Confounded/IMG_4214(1).JPG | border_contact |
| Disease_Confounded/IMG_4215(1).JPG | border_contact |
| Disease_Confounded/IMG_4218(1).JPG | border_contact |
| Disease_Confounded/IMG_4219(1).JPG | border_contact |
| Healthy/healthy_001.PNG | border_contact;foreground_too_large |
| Healthy/healthy_002.JPG | border_contact |
| Healthy/healthy_003.JPG | border_contact |
| Healthy/healthy_005.JPG | border_contact;foreground_too_large |
| Moderate_Water_Stress/moderate_001.JPG | border_contact |
| Moderate_Water_Stress/moderate_002.JPG | border_contact;foreground_too_large |
| Moderate_Water_Stress/moderate_003.JPG | border_contact |
| Moderate_Water_Stress/moderate_004.JPG | border_contact;foreground_too_large |
| Moderate_Water_Stress/moderate_005.JPG | border_contact;foreground_too_large |
| Moderate_Water_Stress/moderate_006.WEBP | border_contact |
| Moderate_Water_Stress/moderate_007.JPG | border_contact;foreground_too_large |
| Moderate_Water_Stress/moderate_008.JPG | border_contact;foreground_too_large |
| Moderate_Water_Stress/moderate_009.JPG | border_contact |
| Severe_Water_Stress/severe_001.JPG | border_contact |
| Severe_Water_Stress/severe_002.JPG | border_contact |
| Severe_Water_Stress/severe_003.JPG | border_contact |
| Severe_Water_Stress/severe_004.JPG | border_contact |
| Severe_Water_Stress/severe_005.JPG | border_contact |
| Severe_Water_Stress/severe_006.JPG | border_contact |
| Severe_Water_Stress/severe_007.JPG | border_contact;foreground_too_large |
| Severe_Water_Stress/severe_008.JPG | border_contact |
| Severe_Water_Stress/severe_009.JPG | border_contact |
| Severe_Water_Stress/severe_010.JPG | border_contact;foreground_too_large |
| Severe_Water_Stress/severe_011.JPG | border_contact |

Byte-identical aliases are listed in the CSV. Distinct photography sessions/plants are unknown. Border rejection may be strict for isolated leaves whose stems touch the frame; review the images rather than changing thresholds solely to increase acceptance.
