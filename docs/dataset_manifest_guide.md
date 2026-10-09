# Dataset manifest guide

`dataset_manifest.csv` has one row per image file (56 rows, including byte-identical copies). The `filename` is relative to `dataset/`. `class_label` is copied from the current folder and is **not independently verified**.

Computed fields:

- `sha256`, `duplicate_of`, original `width` and `height`.
- `gate_status` and `foreground_fraction`: copied by hash from the Stage 3 report, after verifying the unchanged gate reproduces them.
- `border_contact_fraction`: newly measured from the same standardized Stage 3 selected mask. The Stage 3 CSV did not previously contain this measurement. It is the sum of Euclidean contour segment lengths lying entirely along a frame edge divided by total closed external contour length. A single-vertex contact has zero length; an absent/zero-length contour is assigned zero. This diagnostic is distinct from Boolean border contact and does **not** change any rejection rule.
- `gate_reason_codes`: all unchanged gate reasons, separated by semicolons.

Fill these manually: `plant_id`, `leaf_id`, `session_id`, `source` (`own photo` / `web` / `stock`), `device`, `background_type`, `label_method` and `notes`. Stable plant IDs must span classes and sessions. Blank means unknown, not a unique new plant. Do not put “unknown” or “N/A” in identity fields as if those strings were genuine shared IDs. Never infer identity from the image filename. Record original URLs/licensing/provenance in notes for externally sourced images.

The builder/evaluator verify file paths, hashes and label/folder consistency, report cross-label duplicate conflicts, deduplicate by hash, and recompute the gate. Editing `gate_status` to “pass” cannot admit a rejected image. They do not trust a `duplicate_of` annotation instead of checking actual hashes. Conflicting plant/leaf/session IDs on exact copies require human resolution; a populated identity on one copy can supply that identity to an otherwise blank exact copy.

Grouping prefers plant IDs. Where a plant ID is missing, a session ID is used. If that session also contains identified plants, the records are conservatively linked; connected plant/session components remain together. A plant always remains together across sessions. If all plants are known, different plants in one session can remain separate; plant-level grouping does not establish independence from session/environment effects. Session-only grouping cannot exclude an unknown plant recurring in another session. The report states these limitations. Missing both IDs blocks grouped evaluation rather than turning filenames into fake plants.

`dataset_manifest.py` never overwrites an existing manifest or its metadata sidecar. Preserve human edits. The initial generator validates against the fixed Stage 3 snapshot; it deliberately refuses added/changed images absent from that snapshot. For an expanded collection, first generate a new gate report, then call `create_manifest(dataset_root, new_manifest_path, new_report_path)` and reconcile previous annotations by verified identity/hash. Do not overwrite `dataset/` originals or historical reports.

Current readiness: 11 accepted unique images, 4 Healthy and 7 Disease Confounded; no Moderate/Severe accepted images and no known groups. `evaluate_grouped.py` records an explicit blocked report with no fitted models or folds. The optional `--legacy-development-only` mode is separate and must never be reported as performance.
