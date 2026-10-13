# Point-guided SAM for the target plant

This optional local preprocessing stage uses Meta's original SAM ViT-B. It does
not recognize tomato plants or diagnose stress. Keep/exclude points select a
region; the existing colour-based foliage extractor then operates inside that
region. Incorrect masks and missing leaves still require review.

## Install on Windows

For a fresh clone with the active classifier, use [team_setup.md](team_setup.md)
and `setup_project.py` with **CPython 3.13.15**. The commands below add only SAM
to an already compatible application environment. The combined reproducible
environment lock is `requirements-windows-sam.lock.txt`.

From the repository in PowerShell, with the existing application environment:

```powershell
.\.venv-py313\Scripts\python.exe -m pip install -r requirements-sam.txt
.\.venv-py313\Scripts\python.exe setup_sam.py
.\.venv-py313\Scripts\python.exe -m pip check
.\.venv-py313\Scripts\python.exe run_wholeplant.py
```

Git is needed for the pinned official SAM source. PyTorch 2.7.1 CPU and torchvision
0.22.1 CPU are specified for Python 3.13 on Windows; this does not require an NVIDIA
GPU. The checkpoint download is about 375 MB, and the PyTorch wheel is about
216 MB. Both are installed locally; web requests never download models. The first
segmentation loads the model into memory. CPU inference may take tens of seconds
depending on hardware. The model loads once per process; requests are serialized
with a busy response rather than mixing image embeddings across requests.

The default checkpoint is `models/sam_vit_b_01ec64.pth` (excluded from Git).
`LECOR_SAM_CHECKPOINT` can specify a trusted local ViT-B checkpoint;
`LECOR_SAM_DEVICE` defaults to `cpu`. A CUDA device needs a separately compatible
PyTorch CUDA installation. SAM is optional: a missing checkpoint/dependency is
reported in the UI and does not stop the existing crop/classifier workflow.

## Use

1. Upload a JPEG, PNG or WebP. Apply the target-plant crop if useful.
2. In the SAM canvas, select **Keep plant**, then mark several target leaves.
3. Select **Exclude background**, then mark trees/other plants. Numbered marks are
   green for Keep and red for Exclude. Keyboard users can enter X/Y percentages.
4. Click **Generate SAM mask**. Review the green overlay and grey-background image.
   Add marks and regenerate if the mask misses leaves or includes background;
   Undo removes the latest mark. A mask contradicting a point is refused.
5. Click **Use reviewed plant region**, then **Preview foliage mask**, then
   **Analyze image**. Analysis is blocked while a changed selection awaits review.
6. **Clear SAM / use crop** returns to the original applied crop. Applying another
   crop, choosing another image, or Reset clears all SAM points and results.

SAM can select only a leaf rather than the entire plant, especially when foliage
is disconnected or occluded. It can also merge foreground/background. Inspect the
whole overlay and add points; a mask passing point checks is not proof that every
target leaf was retained or every tree leaf excluded. There is no automatic
tomato detection or full-plant completeness guarantee.

Images are aspect-preservingly reduced to a maximum side of 800 pixels before SAM.
The selected region's original colours are preserved; pixels outside it are set
to neutral grey (RGB 220), which the existing foliage colour rules exclude.
The classifier, its 24-feature contract and its deployed model are unchanged.
Consequently, predictions on SAM-prepared inputs are exploratory and can differ
from predictions on the original training distribution; improved classification
accuracy has not been established.

If the page says **Model ran; no confident classification**, the provisional mask
checks passed and inference completed. The highest model probability was below
the 0.60 threshold. The page shows all model scores for inspection while withholding
a class and stress indicator. Review the mask if it is wrong; if it is satisfactory,
repeated clicking is not a remedy for classifier uncertainty. Prepare labelled
training images consistently with SAM and evaluate on independent plants. The
**Review selected plant** button preserves your current image and marks.

## Prepare training inputs consistently

Download **prepared training image** for the exact classifier input, **SAM mask**
for the binary selection, and **selection record** for prompts and original name.
Rename each export uniquely. Store only prepared images in a separate dataset
with the same four class folders. Store original photos, masks and JSON records
outside the training image folders, linking them by unique filename. Preserve real
plant/session IDs and independent label evidence. Do not treat original and
prepared versions of the same photo as independent examples.

```powershell
.\.venv-py313\Scripts\python.exe train_wholeplant.py --dataset dataset_wholeplant_sam --development-only
```

Review the resulting manifest and development report, then update the deployment
pointer to the reviewed new artifact. Do not invent masks for the existing dataset
or assume the same points select the same plant in different photos. Reserve
independent plants for evaluation with the same SAM preparation protocol.

## Software verification

```powershell
.\.venv-py313\Scripts\python.exe -m pytest tests/test_sam.py tests/test_wholeplant.py -q
node tests/frontend_sam.cjs
node tests/frontend_stage1.cjs
.\.venv-py313\Scripts\python.exe verify_sam.py
```

The first tests use a controlled predictor to check geometry, background exclusion,
point validation and error handling. `verify_sam.py` uses the actual checkpoint and
Flask endpoints on a known repository photo; it saves PNG evidence and a JSON report
under `reports/sam_smoke_*`. Its prompt selects a leaf for a software smoke test,
not an assessment of whole-plant segmentation quality or classifier accuracy.

## References

- [Meta SAM source and model checkpoint](https://github.com/facebookresearch/segment-anything)
- [Pinned SAM source revision](https://github.com/facebookresearch/segment-anything/commit/6fdee8f2727f4506cfbbe553e23b895e27956588)
- [PyTorch CPU installation](https://pytorch.org/get-started/previous-versions/)
