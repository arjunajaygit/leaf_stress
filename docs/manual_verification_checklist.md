# Manual device verification checklist

Status: **needs manual verification on device**. No items below have been marked passed by automated tests. No validated classifier exists yet; classification results cannot be presented as performance.

Tester: __________  Date: __________  Commit: __________  Device/OS: __________
Browser and version: __________  URL/deployment mode: __________

Use `LECOR_PREVIEW_ONLY=true python app.py` in the recreated environment. Phone access requires an intentional LAN/HTTPS setup or a server accessible from that device; localhost on a phone is the phone itself. Record the actual setup. Do not describe a developer's local desktop test as phone evidence.

## Camera — REQ-001

- [ ] Phone: open the upload picker; record whether “Take Photo” is offered, permissions work and the captured image reaches the preview.
- [ ] Phone: verify orientation, full-leaf framing, cancellation, retake and large-photo handling.
- [ ] Laptop: check whether camera capture is actually available. The application has no explicit webcam/getUserMedia path; if the picker only selects files, mark webcam capture **not implemented**, not passed.
- [ ] If webcam support is later added, verify permissions, denied permission, camera unavailable, capture and upload on the actual laptop.

## Browser matrix

Complete independently; do not copy one browser's results to others.

| Browser/version | Device/OS | Upload | Mask/rejection | Preview notice/no class | Reset/error | Keyboard | Print | Pass/fail/not tested + evidence |
|---|---|---|---|---|---|---|---|---|
| Chrome | | | | | | | | |
| Firefox | | | | | | | | |
| Safari | | | | | | | | |
| Edge | | | | | | | | |

## Keyboard and accessibility

- [ ] Tab to the upload control; ensure focus is visible and Enter/Space opens the picker without double activation.
- [ ] Cancel picker; focus and subsequent selection remain usable.
- [ ] Scan/Reset/tabs/print are operable by keyboard; Reset stays disabled during a request.
- [ ] Read errors without relying only on color; inspect zoom at 200%, mobile layout and, if available, screen-reader labels.
- [ ] Start a new upload or cause an error; confirm old masks, explanations and classifications do not remain visible.

## Printing/PDF — needs manual verification on device

- [ ] Use “Print preview” and save a PDF; retain the actual file as evidence.
- [ ] Confirm the preview-only/no-validated-classifier notice, mask, rejection reasons and pipeline images are legible and not clipped.
- [ ] Confirm all tabs' relevant content appears, including limitations; no blank/stale image is mistaken for a result.
- [ ] Check page breaks, colors/contrast and images with browser background-print options on/off.
- [ ] **Deferred until a compatible model exists:** use a clearly labelled software fixture if testing layout only; verify every result, probability, explanation, advisory and limitation section appears. Never present fixture results as a real classification demonstration.

## Honest demonstration

- [ ] Start in explicit preview-only mode; notice is visible before and after uploading.
- [ ] Passing image shows masks/pipeline and “classification unavailable”; no class, probability or score appears.
- [ ] Rejected/blank/clipped images show a readable reason and no class/score.
- [ ] Wrong file/corrupt image/oversized upload produces a readable error and cleared results.
- [ ] Explain remaining dataset/label/group gaps and the difference between software tests and scientific performance.

## Observations and acceptance

| Item | Expected behavior | Observed result | Pass/fail/not tested | Screenshot/PDF/reference | Follow-up |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

Course evaluator: confirm which implemented services count toward the minimum three-service requirement. Camera availability, UI design, image effects and temporary storage must be assessed as actually demonstrated; approval: __________.
