# PCOS Detection via Ultrasound Image Classification

A deep learning pipeline designed to detect Polycystic Ovary Syndrome (PCOS) from ultrasound imaging data. The project implements a custom residual convolutional network (ResNet architecture) in PyTorch to classify ultrasound scans based on whether polycystic ovarian morphology is **visible** or **not visible**.

---

## Overview

Polycystic Ovary Syndrome (PCOS) is a widespread endocrine disorder characterized by hormonal imbalances and ovarian cysts detectable via ultrasound. This repository provides end-to-end workflows for:

* Parsing image metadata and labels from Excel (`.xlsx`) and CSV (`.csv`) annotations.
* Preprocessing and augmenting transvaginal/pelvic ultrasound scans.
* Training a custom 4-stage Residual Network from scratch using PyTorch.
* Evaluating model performance with accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.

---

## Repository Structure

```text
radio_heads/
├── PCOS_CNN.ipynb            # Interactive notebook for exploration and baseline CNN/ResNet prototyping
├── Kaggle_resnet_code.ipynb  # Kaggle-adapted pipeline and training workflow
├── pcos_zenedo.py            # Primary standalone training, evaluation, and logging script
├── confusion_matrix.png      # Generated test-set confusion matrix visualization
├── requirements.txt          # Python dependencies
└── old_train/                # Legacy training scripts and experimental checkpoints

```

---

## Model Architecture

The classification network is built around standard residual learning blocks to alleviate vanishing gradients during feature extraction:

* **Input Layer:** $224 \times 224 \times 3$ normalized RGB ultrasound images.
* **Stem:** $7 \times 7$ Conv2D (stride 2, padding 3, 64 channels) + BatchNorm2d + ReLU + $3 \times 3$ MaxPool (stride 2).
* **Residual Stages:**
* **Stage 1:** 2 Residual Blocks (64 channels, stride 1)
* **Stage 2:** 2 Residual Blocks (128 channels, stride 2 downsampling)
* **Stage 3:** 2 Residual Blocks (256 channels, stride 2 downsampling)
* **Stage 4:** 2 Residual Blocks (512 channels, stride 2 downsampling)


* **Classifier Head:** Adaptive Average Pooling (`1x1`) $\rightarrow$ Flatten $\rightarrow$ Fully-connected layer ($512 \rightarrow 2$ logits).
* **Optimization:** Adam Optimizer ($\text{lr} = 0.001$) with Cross-Entropy Loss and validation F1-based early stopping.

---

## Dataset & Directory Setup

The dataset loader (`ExcelLabelDataset`) expects ultrasound image folders paired with tabular label files (`.xlsx` or `.csv`).

### Expected Directory Layout

```text
data/
├── train/
│   ├── class_label.xlsx (or class_label.csv)
│   └── images/
│       ├── img_001.jpg
│       └── ...
└── test/
    ├── class_label.csv (or class_label.xlsx)
    └── images/
        ├── img_101.jpg
        └── ...

```

### Label File Format

The label file must contain at minimum:

* `imagePath`: Relative path to the image file from the split root (e.g., `images/sample1.jpg`).
* `Class label (whether polycsytic ovary is visible or not visible)`: Text labels mapped to binary classes:
* `Visible` $\rightarrow$ `1` (PCOS positive)
* `Not-visible` $\rightarrow$ `0` (PCOS negative / Normal)



---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/HopeBestWorld/radio_heads.git
cd radio_heads

```

### 2. Environment Configuration

Create an isolated virtual environment with Python 3.12:

```bash
python3 -m venv radio_env
source radio_env/bin/activate

```

### 3. Install PyTorch and Dependencies

Install CUDA-enabled PyTorch (adjust CUDA version to your hardware):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

```

---

## Usage

### Training and Evaluation via Script

Run the automated pipeline with early stopping and test evaluation:

```bash
python pcos_zenedo.py

```

* Monitors validation F1-score across epochs.
* Automatically saves model weights to `best_model.pkl`.
* Triggers early stopping after 5 consecutive epochs without F1 improvement.
* Exports `confusion_matrix.png` upon completing test evaluation.

### Running in Jupyter Notebook

To run interactively in JupyterLab or VS Code:

```bash
jupyter lab

```

Open `pcos_zenedo.py` or inspect `PCOS_CNN.ipynb` to step through the data loader, transforms, and per-epoch metrics.

---

## Results & Performance

Evaluation on the test dataset:

| Metric | Score |
| --- | --- |
| **Total Test Samples** | 1,364 |
| **Accuracy** | 78.30% |
| **True Negatives (`not visible`)** | 605 |
| **False Positives** | 223 |
| **False Negatives** | 73 |
| **True Positives (`visible`)** | 463 |
| **Sensitivity / Recall (`visible`)** | 86.38% |

The trained checkpoint shows strong clinical sensitivity ($86.4\%$) on identifying positive PCOS scans, minimizing missed diagnoses. Full classification details and class-level precision/recall curves can be inspected in `confusion_matrix.png`.
