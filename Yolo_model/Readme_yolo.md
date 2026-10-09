# YOLO26 Skin Disease Classification

A Google Colab notebook for preparing a skin-disease image dataset,
auditing image quality and duplicate files, training a pretrained YOLO26
classification model, reviewing validation performance, and exporting
the trained model to ONNX and LiteRT/TFLite-compatible formats.

> **Disclaimer:** This is an educational machine-learning project, not a
> medical device. Predictions must not be used to diagnose, rule out, or
> treat skin conditions. Any real-world clinical use requires
> appropriate clinical validation, privacy review, regulatory
> assessment, and oversight by qualified professionals.

## Project overview

The notebook is named `SDD_YOLO26_cls_v2.ipynb`. It uses Ultralytics
YOLO26 **classification** (`yolo26s-cls.pt`) rather than an
object-detection model. It predicts one class for an input image from a
set of 15 skin-condition categories.

The workflow includes:

1.  Mounting Google Drive and setting a persistent output directory.
2.  Installing Python dependencies and checking whether a GPU is
    available.
3.  Downloading the dataset through `kagglehub`.
4.  Inspecting image paths, class names, class counts, and source split
    names.
5.  Checking image integrity and image dimensions.
6.  Computing MD5 hashes to identify exact duplicate files and possible
    cross-split leakage.
7.  Creating a YOLO classification folder structure for training and
    validation.
8.  Fine-tuning a pretrained YOLO26s classification model.
9.  Reviewing training logs, plots, confusion matrices, and validation
    image predictions.
10. Exporting the trained model to ONNX and attempting a LiteRT/TFLite
    export.

## Dataset

The notebook downloads the dataset using:

``` python
import kagglehub

dataset_path = kagglehub.dataset_download(
    "mgmitesh/skin-disease-detection-dataset"
)
```

Dataset source: [Kaggle --- Skin Disease Detection
Dataset](https://www.kaggle.com/datasets/mgmitesh/skin-disease-detection-dataset)

The notebook discovers image files recursively. Supported extensions are
`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, and `.tiff`.

### Classes

The notebook identifies 15 classes:

1.  Acne
2.  Actinic Keratosis
3.  Basal Cell Carcinoma
4.  Chickenpox
5.  Dermato Fibroma
6.  Dyshidrotic Eczema
7.  Melanoma
8.  Nail Fungus
9.  Nevus
10. Normal Skin
11. Pigmented Benign Keratosis
12. Ringworm
13. Seborrheic Keratosis
14. Squamous Cell Carcinoma
15. Vascular Lesion

### Dataset checks recorded in the notebook

The saved notebook outputs report:

-   **48,233** image files found after integrity checking.
-   **48,233** images marked valid; **0** corrupt images reported.
-   **15** classes.
-   Largest class: **3,978** images.
-   Smallest class: **2,909** images.
-   Largest-to-smallest class ratio: **1.37**.
-   **180** exact-duplicate hash groups, involving **369** images.
-   **21** exact-duplicate groups were reported as appearing across more
    than one source split.

Image dimensions vary. The recorded summary reports a mean width of
approximately 415 pixels and mean height of approximately 354 pixels;
the model is trained with images resized to 224 × 224.

### Important data-splitting caveat

The notebook infers split and class labels from folder names, then
copies only records labelled `train` and `val` into the prepared
dataset. It does not create a separate test set. The saved output lists
**46,334 training images** and **1,899 validation images**.

The displayed per-class counts in the prepared validation split are
highly uneven (for example, 767 Nevus images but only 3 Seborrheic
Keratosis images). The notebook also detects cross-split exact
duplicates but does not remove them or resolve the leakage. These issues
can make the overall validation metrics unreliable or hide poor
performance on underrepresented classes.

**Before treating the results as a trustworthy benchmark, inspect the
source dataset's directory structure, verify that class/split parsing is
correct, remove or group duplicate/near-duplicate images appropriately,
and create a representative held-out test set.** If patient identifiers
exist, split by patient rather than by image. The README does not claim
that these checks have already been resolved.

## Model and training configuration

The notebook loads a pretrained classification checkpoint:

``` python
from ultralytics import YOLO

MODEL_NAME = "yolo26s-cls.pt"
model = YOLO(MODEL_NAME)
```

The recorded training call uses:

  Setting                         Value
  ------------------------------- -------------------------------------
  Model                           `yolo26s-cls.pt`
  Task                            Image classification
  Pretrained weights              Enabled
  Epochs                          40
  Image size                      224 × 224
  Batch size                      32
  Optimizer                       AdamW
  Initial learning rate (`lr0`)   0.001
  Weight decay                    0.0005
  Early-stopping patience         10
  Device                          GPU 0
  Plots                           Enabled
  Output project                  Google Drive `yolo_nl_dl/yolo_runs`
  Run name                        `baseline_224`

The notebook output records Ultralytics 8.4.174, Python 3.13.15, PyTorch
2.11.0+cu130, and a Tesla T4 GPU for the training run. Runtime and
package versions may differ when rerun.

### Recorded training outcome

The notebook output reports that all 40 epochs completed in
approximately **3.920 hours**. During final validation of `best.pt`, the
saved log reports:

-   **Top-1 accuracy:** 0.902 (90.2%)
-   **Top-5 accuracy:** 0.988 (98.8%)

These are the notebook's recorded validation results, not independent
test-set results. Because the notebook has no separate test split and
has the split/duplicate concerns described above, they should not be
interpreted as evidence of clinical accuracy or real-world performance.

## Output locations

Google Drive is mounted at `/content/drive`, and the notebook sets:

``` python
OUTPUT_ROOT = Path("/content/drive/MyDrive/yolo_nl_dl")
```

The training run is saved under:

``` text
My Drive/
└── yolo_nl_dl/
    └── yolo_runs/
        └── baseline_224/
            ├── args.yaml
            ├── results.csv
            ├── results.png
            ├── confusion_matrix.png
            ├── confusion_matrix_normalized.png
            ├── train_batch*.jpg
            ├── val_batch*_labels.jpg
            ├── val_batch*_pred.jpg
            └── weights/
                ├── best.pt
                ├── last.pt
                └── best.onnx   # after successful ONNX export
```

The files shown are the outputs listed or expected from the notebook's
recorded run; actual files can vary by run and Ultralytics version.

### Key artifacts

-   `weights/best.pt`: checkpoint selected as best by the
    training/validation process.
-   `weights/last.pt`: checkpoint from the last completed epoch.
-   `results.csv`: per-epoch training/validation metrics and
    learning-rate values.
-   `results.png`: Ultralytics training-results summary plot.
-   `confusion_matrix.png`: confusion matrix.
-   `confusion_matrix_normalized.png`: normalized confusion matrix.
-   `args.yaml`: training configuration recorded by Ultralytics.
-   `train_batch*.jpg`: example training batches.
-   `val_batch*_labels.jpg`: validation examples with ground-truth
    labels.
-   `val_batch*_pred.jpg`: validation examples with model predictions.
-   `weights/best.onnx`: exported ONNX model when the export cell
    succeeds.

The prepared dataset is built at `/content/skin_yolo_dataset`, which is
**temporary Colab runtime storage**, not the persistent Google Drive
output folder. The source dataset is downloaded into the KaggleHub
cache. If the runtime is deleted, those temporary files may need to be
recreated; the model and run outputs are directed to Drive.

## Running the notebook in Google Colab

1.  Upload/open `SDD_YOLO26_cls_v2.ipynb` in Google Colab.
2.  Select a GPU runtime if available.
3.  Run the Drive-mount cell and authorize access.
4.  Run the dependency installation and imports.
5.  Run the dataset download, inspection, integrity, and
    duplicate-analysis cells.
6.  Run the dataset preparation cells.
7.  Run the training cell. The recorded configuration took about 3.9
    hours on a Tesla T4; actual time depends on hardware and data
    throughput.
8.  Review the model checkpoint, CSV, plots, and confusion matrices in
    `My Drive/yolo_nl_dl/yolo_runs/baseline_224`.
9.  Run the export cells only after confirming `best.pt` exists.

The notebook contains `!pip install -U ultralytics` near the end before
export. Installing/upgrading packages midway through a Colab session can
change the environment and may require a runtime restart if dependency
warnings appear. For repeatable experiments, install a tested set of
versions near the start and record the environment.

## Exporting the trained model

### ONNX

The notebook exports ONNX with:

``` python
from ultralytics import YOLO

model = YOLO(
    "/content/drive/MyDrive/yolo_nl_dl/yolo_runs/baseline_224/weights/best.pt"
)

onnx_path = model.export(format="onnx")
print("ONNX model saved at:", onnx_path)
```

The saved notebook output reports a successful ONNX export at:

``` text
/content/drive/MyDrive/yolo_nl_dl/yolo_runs/baseline_224/weights/best.onnx
```

The recorded ONNX file size is approximately 20.8 MB. Size may vary with
exporter versions and options.

### LiteRT / TFLite

The notebook also calls:

``` python
model.export(format="tflite")
```

The saved output warns that `tflite` is deprecated in the installed
Ultralytics version and is replaced by the unified `litert` format.
Check the current Ultralytics export documentation and confirm the
resulting artifact and runtime compatibility before using it in an
application. The notebook output included the start of the LiteRT
export, but does not clearly establish a final successful artifact path.

## Dependencies

The notebook installs:

``` bash
pip install -U ultralytics
pip install kagglehub scikit-learn seaborn imagehash
```

It also uses Python libraries including:

-   PyTorch
-   NumPy
-   pandas
-   Matplotlib
-   Seaborn
-   Pillow
-   scikit-learn
-   ImageHash
-   Ultralytics
-   KaggleHub

Some are installed as dependencies of the listed packages. For
reproducibility, pin and test exact package versions rather than relying
on `-U` for every run.

## Repository recommendations

For a GitHub repository, include:

-   `SDD_YOLO26_cls_v2.ipynb`
-   `README.md`
-   A `requirements.txt` with tested package versions
-   Selected plots and evaluation artifacts, if useful and permitted

Avoid committing the full dataset unless its license permits
redistribution. The `.pt` and `.onnx` model files may be large; use Git
LFS or link to a suitable external download location if distribution is
allowed. Never commit credentials, private data, or access tokens.

## Limitations and responsible use

-   The notebook reports validation metrics but does not evaluate on a
    separate held-out test set.
-   The source split parsing and the extreme imbalance in the displayed
    validation counts need review.
-   Exact duplicates across splits were detected but not removed in the
    notebook.
-   No patient-level split or independent clinical validation is
    demonstrated.
-   A high Top-1 or Top-5 score alone does not establish reliability for
    diagnosis.
-   Dataset licensing, image provenance, demographic representation, and
    potential bias should be reviewed before any redistribution or
    real-world use.

## Citation and attribution

-   Dataset: [mgmitesh/skin-disease-detection-dataset on
    Kaggle](https://www.kaggle.com/datasets/mgmitesh/skin-disease-detection-dataset)
-   Model and training framework:
    [Ultralytics](https://github.com/ultralytics/ultralytics)
-   Notebook: `SDD_YOLO26_cls_v2.ipynb`

Please check the dataset's license and the applicable Ultralytics
licensing terms before redistributing data or deploying the model.
