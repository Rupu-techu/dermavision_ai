# DermaVision AI — Module 3 (Subhankar Part)

**Skin pigmentation detection, analysis and segmentation (local GPU version)**

Module 3 of DermaVision AI is a standalone notebook, `module-3_local_gpu.ipynb`, that runs three computer-vision tasks on a skin-condition image dataset. It does not retrain the YOLO classification baseline.

| # | Task | Technique | Output |
|---|------|-----------|--------|
| 1 | Skin-condition classification | Transfer learning: ResNet50, EfficientNet-B0, MobileNetV2 | Trained classifiers, metrics, confusion matrices |
| 2 | Localising pigmented regions | YOLO detection on pseudo bounding boxes | Detector weights, mAP, predictions |
| 3 | Pigmentation percentage | U-Net segmentation on pseudo masks | Segmentation model, per-image % report |

> **Disclaimer:** educational / research project. Not a medical device; not for diagnosis.

## Contents

1. [Pipeline overview](#pipeline-overview)
2. [Environment and hardware handling](#environment-and-hardware-handling)
3. [Dataset preparation](#dataset-preparation)
4. [Part 1 — Transfer learning classifiers](#part-1--transfer-learning-classifiers)
5. [Pseudo-label generation](#pseudo-label-generation)
6. [Part 2 — YOLO detection](#part-2--yolo-detection)
7. [Part 3 — U-Net segmentation](#part-3--u-net-segmentation)
8. [Pigmentation percentage and report](#pigmentation-percentage-and-report)
9. [Setup and usage](#setup-and-usage)
10. [Output layout](#output-layout)
11. [Hyperparameter reference](#hyperparameter-reference)
12. [Results](#results)
13. [Limitations](#limitations)
14. [Troubleshooting](#troubleshooting)
15. [Author](#author)

---

## Pipeline overview

```
Kaggle dataset ──► cleaned train/val class folders ──┬─► Part 1: ResNet50 / EfficientNet-B0 / MobileNetV2 ─► metrics, confusion matrices
                                                     │
                                                     ├─► pseudo boxes ─► Part 2: YOLO detection ─► mAP, predictions
                                                     │
                                                     └─► pseudo masks ─► Part 3: U-Net ─► pigmentation % per image (CSV)
```

---

## Environment and hardware handling

- **Reproducibility:** seed 42 for `random`, NumPy and PyTorch (including CUDA).
- **Device selection:** CUDA, then Apple MPS, then CPU.
- **Mixed precision (AMP):** enabled automatically on CUDA; `cudnn.benchmark` is turned on.
- **DataLoader workers:** 0 on Windows (multiprocessing in notebooks is fragile there), otherwise `min(4, cpu_count)`. `pin_memory` is enabled on CUDA.
- **Batch sizes scaled to GPU memory:**

| GPU VRAM | Classifier batch | U-Net batch |
|----------|------------------|-------------|
| under 5 GB | 16 | 8 |
| 5–7 GB | 32 | 16 |
| 7 GB and above | 64 | 16 |
| no GPU | 16 | 8 |

- **YOLO batch** uses the classifier batch size, capped at 32 for the 416 px detection run.
- **Guard cell:** checks that a CUDA build of PyTorch is installed before `ultralytics` is installed, so `pip` does not replace it with a CPU build.
- Weights & Biases logging is disabled.

---

## Dataset preparation

Source: Kaggle `mgmitesh/skin-disease-detection-dataset`, downloaded with `kagglehub`.

1. Walk every file; keep supported image types (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`).
2. Detect the split from the folder path (`train`, `val` / `valid`, `test`) and the class from the folder that follows the split name.
3. Verify each image with PIL (`verify()`); corrupt files are counted and skipped. Truncated images are tolerated.
4. Copy the images to `skin_yolo_dataset/{train,val}/<class>/`. Duplicate filenames are renamed.
5. The step is skipped automatically if the split already exists.

Class names and `NUM_CLASSES` are read from the `train` folder; train and validation class order is checked to match.

---

## Part 1 — Transfer learning classifiers

**Data pipeline (224×224, `ImageFolder`)**

| | Transforms |
|--|-----------|
| Train | Resize, random horizontal flip, random rotation (15°), colour jitter (brightness / contrast / saturation 0.2), ImageNet normalisation |
| Validation | Resize, ImageNet normalisation |

**Models** (ImageNet-pretrained, final layer replaced for `NUM_CLASSES`; all layers are fine-tuned)

| Architecture | Weights | Replaced layer |
|--------------|---------|----------------|
| ResNet50 | `IMAGENET1K_V2` | `fc` |
| EfficientNet-B0 | `IMAGENET1K_V1` | `classifier[1]` |
| MobileNetV2 | `IMAGENET1K_V2` | `classifier[1]` |

**Training**
- Loss: cross-entropy. Optimiser: AdamW (lr 1e-4, weight decay 1e-4). Schedule: cosine annealing over the epoch count.
- Up to 15 epochs, early stopping with patience 5 on validation accuracy; the best weights are restored and saved to `transfer_learning_runs/<arch>_best.pt`.
- AMP with `GradScaler` on CUDA.
- Models are trained one after another with memory cleanup in between. A CUDA out-of-memory error in one model is caught and reported, and training continues with the next.
- Peak GPU memory is printed after each model.

**Evaluation**
- Accuracy, balanced accuracy, macro precision, macro recall and macro F1 on the validation set, plus a per-class classification report.
- Side-by-side comparison table of all models.
- Confusion matrix heatmap for each model.
- Validation accuracy and loss curves per epoch.
- Optional comparison against an earlier YOLO classification baseline (`yolo_runs/baseline_224/weights/best.pt`) evaluated at 224 px; skipped if the weights are not present.

---

## Pseudo-label generation

The dataset has image-level labels only, so boxes and masks are produced by weak supervision (classical image processing). The same logic drives Part 2 and Part 3.

1. Convert to LAB colour space and take the lightness channel `L`.
2. Light Gaussian blur (7×7) to suppress noise.
3. Estimate a **local skin-tone baseline** with a heavy Gaussian blur (σ ≈ image size / 8), so lighting gradients cancel out.
4. `diff = baseline − blurred`, clipped at 0: only pixels **darker** than their surroundings count, and bright glare is ignored.
5. Otsu threshold on `diff`. If the threshold is below `min_thr` (default 12), the skin is treated as uniform and no region is produced.
6. Morphological opening (5×5) and closing (9×9) to remove specks and fill gaps.

```python
b  = cv2.GaussianBlur(l_channel, (7, 7), 0)
bg = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=max(h, w) / 8)
diff = np.clip(bg.astype(np.float32) - b.astype(np.float32), 0, 255).astype(np.uint8)
thr, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

- **Boxes** (`generate_pseudo_bboxes`): contours with area of at least 1% of the image, up to the 5 largest, returned as `(x, y, w, h)`. If none are found, the detection builder uses a centred fallback box covering 80% of the image.
- **Masks** (`generate_pseudo_mask`): the image is resized to 256×256 and a binary mask is returned. Unreadable files return `(None, None)` and are skipped; uniform skin returns an all-zero mask.

---

## Part 2 — YOLO detection

**Dataset construction (`skin_yolo_detect/`)**
- Only real image files are processed. File names are `<class>__<stem>_<ext>`, so same-name files never collide.
- Labels are written in YOLO format: `class_id x_center y_center width height`, normalised to 0–1. The class id is the disease class of the source image.
- `data.yaml` lists the train and val image folders and the class names.
- The builder reports how many images used the fallback box.

**Training**

| Setting | Value |
|---------|-------|
| Model | `yolo26n.pt` (falls back to `yolo11n.pt` if unavailable) |
| Epochs / patience | 30 / 10 |
| Image size | 416 |
| Optimiser | AdamW, lr0 0.001, weight decay 0.0005 |
| Device / workers / AMP | from the hardware setup |
| Run folder | `yolo_runs/detection_pigmentation` |

The training cell runs pre-flight checks (`data.yaml` and image folders exist), sets the seed, and gives a clear message on CUDA out-of-memory.

**Evaluation and prediction**
- `best.pt` is evaluated with `imgsz` equal to the training size; mAP50 and mAP50-95 are printed.
- Six validation images are predicted (confidence 0.25) and plotted with their boxes.

---

## Part 3 — U-Net segmentation

**Dataset (`skin_seg_dataset/`)**
- Pseudo masks are built at 256×256 and saved as PNG next to the resized image.
- Up to 150 training and 40 validation images per class, chosen randomly with a fixed seed. Set `max_per_class=None` to use everything.
- The builder reports how many masks are empty.

**`SkinSegmentationDataset`**
- Loads the image (RGB) and mask, resizes to 256×256 (nearest-neighbour for masks), random horizontal flip with p = 0.5 on the training set, image scaled to 0–1, mask thresholded at 127.

**Architecture**
- Four encoder levels (32, 64, 128, 256 channels), a 512-channel bottleneck, and four decoder levels with transposed-convolution upsampling and skip connections (concatenation).
- Each block is `DoubleConv`: (3×3 conv → BatchNorm → ReLU) twice.
- A 1×1 convolution produces one logit per pixel. Input size must be divisible by 16 (256 is).

**Loss and metrics**
- Loss = binary cross-entropy with logits + soft Dice loss (per-sample).
- Validation Dice and IoU are computed from sigmoid outputs at a 0.5 threshold; the metric functions take raw logits.

**Training**
- AdamW (lr 1e-3, weight decay 1e-5), cosine annealing over 30 epochs, AMP on CUDA.
- The forward pass runs under autocast; the loss is computed in FP32.
- Shape and finite-loss checks run each batch.
- Early stopping on validation Dice (patience 7). The best checkpoint is saved as a dictionary (`epoch`, `model_state_dict`, `optimizer_state_dict`, `best_val_dice`, `val_loss`, `val_iou`) and reloaded after training.
- Loss, Dice and IoU curves are plotted.

---

## Pigmentation percentage and report

`predict_pigmentation_percentage(image_path, model)`:
1. Resize to 256×256, run the U-Net, apply sigmoid, threshold at 0.5.
2. Build a skin mask with a YCrCb colour range (Cr 133–173, Cb 77–127) followed by morphological opening.
3. **Percentage = pigmented pixels on skin ÷ skin pixels × 100.** If less than 5% of the image looks like skin, or `skin_only=False`, the percentage of the whole image is used instead.

Visualisation: original image with the percentage in the title, and a red overlay of the pigmented region.

**Batch report:** every validation image gets a percentage, saved to `pigmentation_report.csv` (filename, class, percentage). A per-class summary (count, mean, median, max) is saved to `pigmentation_class_summary.csv`. Images that fail are collected and listed instead of stopping the run.

---

## Setup and usage

**Requirements:** Python 3.10+; an NVIDIA GPU is recommended (Apple MPS and CPU also work); Kaggle access for the dataset.

Install a **CUDA build of PyTorch first**:

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Linux / macOS:         source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -U ultralytics imagehash kagglehub seaborn scikit-learn opencv-python pillow pandas matplotlib ipywidgets
```

Choose the CUDA version that matches `nvidia-smi` from <https://pytorch.org/get-started/locally/>.

**Run**
1. Open `module-3_local_gpu.ipynb` in VS Code and select the `.venv` kernel.
2. Set `BASE_DIR` once in the SETUP cell, for example `Path(r"D:\nn")` or `Path("./working").resolve()`.
3. Run all cells from top to bottom. Dataset preparation, label generation, training, evaluation and the final report run in order.

---

## Output layout

```
BASE_DIR/
├── skin_yolo_dataset/                  # cleaned train/val class folders
├── transfer_learning_runs/             # resnet50_best.pt, efficientnet_b0_best.pt, mobilenet_v2_best.pt
├── skin_yolo_detect/                   # YOLO images, labels, data.yaml
├── yolo_runs/detection_pigmentation/   # YOLO run, weights/best.pt
├── skin_seg_dataset/                   # U-Net images and masks
├── unet_pigmentation_best.pt           # best U-Net checkpoint
├── pigmentation_report.csv             # pigmentation % per validation image
└── pigmentation_class_summary.csv      # per-class summary
```

---

## Hyperparameter reference

| Component | Setting | Value |
|-----------|---------|-------|
| Classifiers | Input / epochs / patience | 224 / 15 / 5 |
| | Optimiser | AdamW, lr 1e-4, wd 1e-4, cosine schedule |
| | Augmentation | flip, rotation 15°, colour jitter 0.2 |
| Pseudo labels | `min_thr` | 12 |
| | Minimum box area | 1% of the image |
| | Maximum boxes per image | 5 |
| YOLO | Model / image size / epochs | `yolo26n` / 416 / 30 |
| | Optimiser | AdamW, lr0 0.001, wd 0.0005, patience 10 |
| U-Net | Input / base features | 256 / 32 |
| | Loss | BCE + Dice |
| | Optimiser | AdamW, lr 1e-3, wd 1e-5, cosine schedule |
| | Epochs / patience | 30 / 7 |
| | Samples per class (train / val) | 150 / 40 |
| Percentage | Threshold / min skin ratio | 0.5 / 0.05 |

---

## Results

Fill in after running the notebook.

| Model | Accuracy | Balanced acc. | Macro precision | Macro recall | Macro F1 |
|-------|----------|---------------|-----------------|--------------|----------|
| ResNet50 | | | | | |
| EfficientNet-B0 | | | | | |
| MobileNetV2 | | | | | |

| Task | Metric | Value |
|------|--------|-------|
| YOLO detection | mAP50 / mAP50-95 | |
| U-Net segmentation | Best validation Dice / IoU | |

---

## Limitations

- **Weak supervision.** Boxes and masks come from colour thresholding, not expert annotation. The detector and U-Net learn to imitate this heuristic, so their scores show agreement with the pseudo labels, not clinical accuracy.
- **The pigmentation percentage is an estimate** of dark-region area and depends on lighting, camera and skin tone.
- **The skin mask is a colour-range heuristic** and can fail on some skin tones or backgrounds.
- Replace the pseudo labels with dermatologist annotations before drawing clinical conclusions.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| PyTorch is CPU-only although a GPU exists | Reinstall with the CUDA index URL above, then restart the kernel. |
| `CUDA out of memory` | Lower `BATCH_SIZE`, `SEG_BATCH` or `YOLO_BATCH`, or reduce `YOLO_IMGSZ`. |
| Validation image count is 0 after dataset preparation | The dataset has no `val` folder. Map `"test"` to `"val"` in `detect_split` and re-run. |
| Most pseudo masks or boxes are empty | Lower `min_thr` (about 8) in both pseudo-label functions and rebuild the datasets. |
| Boxes appear on normal skin | Raise `min_thr` (about 16). |
| U-Net Dice stays near 0 | Too many empty masks; lower `min_thr` and rebuild the segmentation dataset. |
| Path errors on `D:\` | Set `BASE_DIR` once in the SETUP cell to a folder that exists on your machine. |

---

## Author

Subhankar Nandi — [github.com/subhankarnandi777](https://github.com/subhankarnandi777)
