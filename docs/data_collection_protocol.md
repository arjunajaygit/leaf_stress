# Practical leaf-image collection protocol

## Suggested minimum targets

These are planning targets, **not statistically derived sample sizes, guarantees of accuracy, or proof of independence**. The evaluator's five-groups-per-class guard only checks whether its nested-CV protocol may be feasible; it is not the collection target.

| Visual category requiring independent labels | Suggested accepted images | Distinct leaves | Plants represented | Acquisition sessions |
|---|---:|---:|---:|---:|
| Healthy | At least 30 | At least 30 | At least 10 | At least 3 |
| Moderate Water Stress | At least 30 | At least 30 | At least 10 | At least 3 |
| Severe Water Stress | At least 30 | At least 30 | At least 10 | At least 3 |
| Disease-like / other confounding symptoms | At least 30 | At least 30 | At least 10 | At least 3 |

Spread each class across plants and sessions, rather than photographing one class on one date/background. Extra views or repeated sessions of a leaf are welcome, but do not count them as additional independent leaves/plants. The same plant may change class over time: preserve its plant ID across classes and sessions and keep all of it in one split. Start with one recorded crop/cultivar and similar leaf maturity. The disease-like set needs documented causes/diversity; image appearance alone cannot confirm disease or distinguish it from drought.

## Capture and label

1. Photograph **one leaf**, fully inside the frame with background margin. Use a plain neutral background and diffuse light. Avoid hard shadows, glare and overlapping leaves. Keep leaf orientation, camera distance and framing comparable across all classes; record distance and deviations in notes.
2. Use stable, globally unique plant and leaf IDs; never restart IDs per class. Record session ID, device, source, cultivar/leaf age, lighting and treatment information. Retain identities on repeat photographs. Record actual acquisition sessions, not upload dates or filenames.
3. Assign labels independently of the RGB/shape features the model uses. Record who labelled the plant, the method, measurements and date. Controlled irrigation history plus direct soil checks and, where feasible, physiological measurements or expert assessment provide evidence; time since watering or yellowing alone does not establish severity. Uncertain samples should remain explicitly unresolved until reviewed; do not guess a four-class label to meet quotas.
4. For web/stock images, document the original source and provenance in notes. Do not fabricate plant/session IDs or count different filenames as independent plants. Unresolvable provenance limits grouped evaluation; new own-photo collections are preferable.
5. Inspect the generated mask and retake failed inputs. Preserve the original image and rejection history. Do not relax thresholds just to meet a target.
6. Reserve **whole plants and later acquisition sessions** for a final test set before development. To test both new-plant and later-session generalization, use new plants captured in later sessions; do not put later photos of training plants into a supposedly independent plant test. Keep final-test files outside the development `dataset/` and manifest until the pipeline is frozen. Touch the final test once; any subsequent tuning makes it development data.

## Collector's log — fill this in

Copy/add rows as needed. Enter unknown explicitly in this log, but leave unverified manifest identity fields blank rather than inventing IDs.

| Filename | Proposed class / unresolved | Plant ID | Leaf ID | Session ID/date | Device | Source | Background / distance | Label method, assessor and evidence | Development or held-out plant/session | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

Complete the corresponding fields in `dataset_manifest.csv` after provenance and labels are reviewed. `source` should be `own photo`, `web` or `stock`. The current manifest deliberately leaves all plant/leaf/session/source/label-method/device/background fields blank. No identity was inferred from similar-looking leaves or filenames.
