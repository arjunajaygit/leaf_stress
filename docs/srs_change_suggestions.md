# Proposed SRS wording changes — Stage 2

Date: 2026-10-08. These are proposed edits only. The supplied PDF has not been modified. The project owner must approve/apply the document changes and update its revision history.

## Version discrepancy

Source filename: `Lecor_SRS_v0.3-4.pdf`. The cover and revision history instead state **Version 0.1**, dated 2026-09-18. The filename alone does not establish the approved document version.

Proposed resolution: confirm the authoritative version with the authors. If the supplied document is accepted as v0.3, publish these revisions as **Version 0.4**, name the document `Lecor_SRS_v0.4.pdf`, and add a revision entry dated on actual approval: “Corrected visual-assessment claims, input-validation status, score interpretation and feature comparisons.” If v0.1 is authoritative, choose the next approved version instead. Do not invent prior approvals or backdate the revision.

## REQ-002 (page 8)

Current: “The system shall display a water-stress classification, a 0–100 stress score, and a plain-language explanation on the web UI.”

Replace with: “The system shall display the predicted visual category, model class probabilities, descriptive feature comparisons, and cautious inspection guidance on the web UI. For supported non-confounded results, it may display an Experimental relative visual-stress indicator as an integer from 0 to 100, clearly identified as an unvalidated heuristic rather than a physiological measurement. The indicator shall not be displayed for Disease Confounded results. The visible label for that class shall be ‘Disease-like or other confounding symptoms suspected’. A limitations note shall remain visible with results.”

## REQ-005 (page 8)

Current: “The system shall detect the leaf's outline and compute shape/curl indicators (such as solidity, compactness, extent, and aspect ratio) from it.”

Replace with: “The system shall compute 2D projected shape descriptors, including solidity, circularity/compactness, extent and aspect ratio, from the segmented outline. The system shall describe these as image-shape measurements, not direct measurements of leaf curl, turgor or water content. Their interpretation shall account for perspective, natural leaf shape and segmentation errors.”

## REQ-008 (page 8)

Current: “The system shall handle invalid input (e.g. no leaf detected, or a file that cannot be read as an image) by displaying an error message asking the user to retake or re-upload the photo.”

Replace with: “The system shall enforce permitted image formats, a request-size limit and a decoded-pixel limit; verify that the submitted file decodes as an image; and return a structured JSON error for invalid input or processing, model or feature-schema failures. The web UI shall clear previous results and display a readable error, asking for another photo when appropriate or operator assistance for server/model failures. Successful decoding shall not be represented as proof that the image contains a leaf.”

Stage 3 extension, not yet implemented: “Inputs failing mask-quality checks shall receive a reason and a retake request without a class or indicator.”

## REQ-009 (page 8)

Current: “The system shall calculate a stress score from 0 to 100 based on the model's prediction and the leaf measurements.”

Replace with: “For non-confounded predictions, the system shall calculate an Experimental relative visual-stress indicator using a documented heuristic based on model probabilities and image measurements. It shall display an integer on a 0–100 scale without a percentage sign, state that the indicator is not a physiological measurement or model confidence, and omit it for Disease Confounded results. Physiological validity shall not be claimed without independent calibration against measured plant water status.”

## REQ-010 (page 8)

Current: “The system shall generate a plain-language explanation of the result by comparing the leaf measurements against typical values.”

Replace with: “The system shall display ‘Feature measurements vs. dataset averages’, comparing available feature values with pooled all-class dataset means and standardized deviations. These comparisons shall not be labelled as contributions to the classifier's prediction, biological causes, or deviations from a healthy reference population. Guidance shall recommend independent inspection and verification without prescribing irrigation quantities, recovery times or chemical treatments.”

Stage 5 may add a separately labelled model-specific explanation; this is not currently implemented.

## Section 1.3.4 — replace the limitations list (page 6)

Proposed exact replacement:

- “Lecor is an academic visual-only prototype trained on a small dataset. The present dataset has 56 image files and 40 byte-unique images. This count describes the supplied dataset, not the number of independent plants.”
- “Plant/leaf/session identifiers, label provenance and independent physiological validation are incomplete. Background and framing vary by class. Reported image-level development cross-validation does not establish generalization to new plants, devices or field conditions.”
- “Image measurements depend on camera settings, lighting, perspective, background and segmentation. The current segmentation works best for one fully visible leaf on a plain contrasting background and can incorrectly include backgrounds, multiple leaves or non-leaf objects.”
- “The system measures RGB/HSV color and 2D projected shape descriptors, not water content, turgor, actual curl angle, or the angle between leaf and stem.”
- “The Experimental relative visual-stress indicator is an unvalidated heuristic, not a physiological water-stress measurement or probability. It is omitted for Disease Confounded predictions.”
- “Disease-like symptoms can overlap with drought, nutrient problems, pests, heat and other causes. Drought and disease may coexist. The system does not establish symptom cause or prescribe treatment.”
- “Color-threshold maps show pixel groups, not confirmed healthy, chlorotic or necrotic tissue. Feature comparisons use pooled dataset averages, not a healthy reference or prediction contributions.”
- “The system is not a replacement for direct soil assessment, laboratory testing or qualified agricultural advice. Early-warning effectiveness has not been validated.”
- “The system checks file format, request size, image dimensions and decodability. These checks do not establish that an image contains a valid single leaf. Automatic mask-quality rejection and low-confidence abstention remain pending Stage 3.”
- “Scan history is not saved to a database. Uploaded files are temporary and are removed after processing. Camera capture, print completeness and cross-browser behavior require manual device verification.”
- “The selected classifier can change after an explicitly reviewed retraining run. New artifacts use versioned filenames; the original data and artifacts are preserved.”

## Contradiction requiring explicit resolution

Section 1.3.4 currently says: “The system does not yet check whether an uploaded file is actually a photo before processing it.” REQ-008 simultaneously requires handling unreadable images and no-leaf inputs. This is a requirements-versus-implementation-status contradiction/ambiguity: error handling after attempted decoding is not the same as checking an upload before processing, and neither proves leaf identity.

Stage 1 now validates permitted formats, dimensions and decoding before feature processing. Replace the outdated limitation with the proposed validation limitation above, and retain the distinction between a readable image and an acceptable leaf image. Do not mark the pending Stage 3 semantic/quality checks complete.

## Consistency edits elsewhere

Use the same corrected terms in the scope, product functions, definitions, figure captions and UI descriptions. Replace direct “curling measurement” claims with “2D projected shape descriptors”; replace diagnostic/pathology implications with visual-category and color-group descriptions. Keep REQ-006's internal class identifier for artifact compatibility, but apply the revised disease display label. The class names do not establish verified disease or physiological status. Do not describe separate model predictions as voting or as cross-validation on an uploaded image.
