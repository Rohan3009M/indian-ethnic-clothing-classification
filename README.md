# Indian Ethnic Clothing Classification

Deep learning image classification for 15 categories of Indian ethnic clothing using PyTorch and transfer learning on the IndoFashion dataset.

## Overview

This repository implements an end-to-end image classification pipeline that covers:

- dataset preparation from the original IndoFashion metadata
- balanced subset creation for manageable experimentation
- transfer learning with multiple pretrained CNN backbones
- GPU training with mixed precision support
- evaluation with classification reports and confusion matrices
- a Streamlit app for quick testing with trained checkpoints

## How I Built This Project

I built this project in a simple pipeline:

1. I started with the IndoFashion dataset and created a balanced subset with 15 classes and 500 images per class.
2. I organized the data into train, validation, and test folders so it could be used directly with PyTorch dataloaders.
3. I used transfer learning with pretrained CNN models and replaced their final classifier layers with a custom classification head.
4. I trained and evaluated multiple models, saved checkpoints and reports, and then compared their final performance.
5. I added a `streamlit.py` app so the trained models can be tested interactively on uploaded images.

This keeps the project focused on a complete but manageable deep learning workflow rather than a very large experimental setup.

## Dataset

Source: https://indofashion.github.io/

The original dataset contains about 106,000 images. This project uses a balanced subset for faster and more consistent experiments.

### Subset Summary

| Item | Value |
| --- | ---: |
| Classes | 15 |
| Images per class | 500 |
| Total images | 7,500 |

### Split Summary

| Split | Images | Per class |
| --- | ---: | ---: |
| Train | 5,250 | 350 |
| Validation | 1,125 | 75 |
| Test | 1,125 | 75 |
| Total | 7,500 | 500 |

### Classes

`blouse`, `dhoti_pants`, `dupattas`, `gowns`, `kurta_men`, `leggings_and_salwars`, `lehenga`, `mojaris_men`, `mojaris_women`, `nehru_jackets`, `palazzos`, `petticoats`, `saree`, `sherwanis`, `women_kurta`

## Project Structure

```text
indian-ethnic-clothing-classification/
|-- data/
|   |-- raw/
|   |   `-- indofashion_dataset/
|   |-- interim/
|   `-- processed/
|       `-- dataset_subset/
|           |-- train/
|           |-- val/
|           `-- test/
|-- outputs/
|   |-- checkpoints/
|   |-- figures/
|   |   |-- confusion_matrices/
|   |   |-- model_comparison/
|   |   `-- training_curves/
|   |-- logs/
|   `-- reports/
|-- scripts/
|-- src/
|   |-- data/
|   |-- evaluation/
|   |-- models/
|   |-- training/
|   `-- utils/
|-- streamlit.py
|-- requirements.txt
`-- README.md
```

## Environment

The project was developed with:

- Python 3.10
- PyTorch with CUDA support
- NVIDIA RTX 4060 Laptop GPU
- CUDA 12.8

Install the listed Python dependencies with:

```bash
pip install -r requirements.txt
```

Note: `torch`, `torchvision`, and `streamlit` may need to be installed separately depending on your environment.

## Workflow

### 1. Prepare the Dataset Subset

Expected raw dataset location:

```text
data/raw/indofashion_dataset
```

Run:

```bash
python scripts/prepare_dataset.py
```

This script:

- reads metadata from the raw dataset JSONL files
- normalizes class names
- samples 500 images per class
- creates `train`, `val`, and `test` splits
- copies images into `data/processed/dataset_subset`
- saves a summary to `data/interim/subset_summary.csv`

### 2. Train a Model

Example:

```bash
python scripts/train.py --model_name mobilenet_v2 --epochs 5
```

Optional frozen-backbone training:

```bash
python scripts/train.py --model_name resnet50 --epochs 5 --freeze_backbone
```

Useful arguments:

- `--model_name`: `resnet50`, `mobilenet_v2`, `efficientnet_b0`, `densenet121`
- `--epochs`: number of training epochs
- `--batch_size`: batch size, default `32`
- `--image_size`: image size, default `224`
- `--lr`: learning rate, default `1e-3`
- `--weight_decay`: weight decay, default `1e-4`
- `--freeze_backbone`: trains only the classification head

Training outputs:

- checkpoints in `outputs/checkpoints`
- training history in `outputs/logs/<model_name>/history.json`
- training curves in `outputs/figures/training_curves`

### 3. Evaluate a Trained Model

Example:

```bash
python scripts/evaluate.py --model_name mobilenet_v2
```

If the model was trained with a frozen backbone, pass:

```bash
python scripts/evaluate.py --model_name resnet50 --freeze_backbone
```

Evaluation outputs:

- classification report in `outputs/reports`
- confusion matrix in `outputs/figures/confusion_matrices`
- metrics JSON in `outputs/reports`

### 4. Compare Models

Run:

```bash
python scripts/evaluate_all_models.py
python scripts/compare_models.py
```

This generates:

- `outputs/reports/model_comparison_summary.csv`
- accuracy, macro F1, and weighted F1 comparison plots in `outputs/figures/model_comparison`

### 5. Test the Model with Streamlit

The `streamlit.py` file provides a simple UI for testing trained models on uploaded clothing images.

Run:

```bash
streamlit run streamlit.py
```

The app lets you:

- choose one of the trained models
- upload an image
- see the predicted class
- view the top prediction probabilities

Make sure the trained checkpoint files are available in `outputs/checkpoints/` before launching the app.

## Models Evaluated

The project compares the following pretrained CNN architectures:

- ResNet50
- MobileNetV2
- EfficientNet-B0
- DenseNet121

Each model replaces the default classifier with a custom head:

```text
Linear -> ReLU -> Dropout -> Linear
```

## Training Setup

- Loss: `CrossEntropyLoss`
- Optimizer: `AdamW`
- Scheduler: `StepLR`
- Mixed precision: enabled during training
- Device selection: automatic CPU/GPU detection
- Checkpointing: best model is saved during training

## Evaluation Metrics

The evaluation pipeline reports:

- accuracy
- macro precision
- macro recall
- macro F1
- weighted F1
- confusion matrix

## Model Performance

Results from `outputs/reports/model_comparison_summary.csv`:

| Model | Test Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| MobileNetV2 | 77.42% | 0.770 | 0.770 |
| DenseNet121 | 77.24% | 0.767 | 0.767 |
| ResNet50 | 73.96% | 0.735 | 0.735 |
| EfficientNet-B0 | 69.51% | 0.685 | 0.685 |

## Best Model

`MobileNetV2` performed best in this experiment.

Reasons it stands out:

- highest test accuracy in the current comparison
- best macro F1 and weighted F1 among tested models
- lightweight architecture with strong generalization

## Outputs

```text
outputs/
|-- checkpoints/
|-- figures/
|   |-- confusion_matrices/
|   |-- model_comparison/
|   `-- training_curves/
|-- logs/
`-- reports/
```

Key generated files include:

- `outputs/reports/*_classification_report.txt`
- `outputs/reports/*_test_metrics.json`
- `outputs/reports/model_comparison_summary.csv`
- `outputs/figures/confusion_matrices/*.png`
- `outputs/figures/training_curves/*.png`
- `outputs/figures/model_comparison/*.png`

## Notes

The dataset is not included in this repository because of its size. Download it from the official IndoFashion source and place it under `data/raw/indofashion_dataset` before running the pipeline.
