<p align="center">

# 🇮🇳 Indian Ethnic Clothing Classification

Deep Learning Image Classification using PyTorch and Transfer Learning  
IndoFashion Dataset

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![GPU](https://img.shields.io/badge/GPU-CUDA%2012.8-green)
![Dataset](https://img.shields.io/badge/Dataset-IndoFashion-orange)

</p>

🇮🇳 Indian Ethnic Clothing Classification using Deep Learning

Deep learning project for classifying Indian ethnic clothing categories using the IndoFashion dataset with transfer learning and PyTorch.

The project builds a complete end-to-end machine learning pipeline, including:

dataset preparation

data augmentation

transfer learning

GPU training

model evaluation

confusion matrix analysis

automated model comparison

📌 Project Overview

Indian ethnic fashion contains a wide variety of garments that are visually similar and complex to classify.

This project builds a deep learning image classification system capable of recognizing 15 categories of Indian ethnic clothing using pretrained convolutional neural networks.

The project focuses on:

transfer learning with pretrained CNNs

comparing multiple architectures

building a reproducible ML pipeline

analyzing classification errors using confusion matrices

🧵 Clothing Categories

The model classifies the following 15 Indian ethnic clothing types:

blouse

dhoti_pants

dupattas

gowns

kurta_men

leggings_and_salwars

lehenga

mojaris_men

mojaris_women

nehru_jackets

palazzos

petticoats

saree

sherwanis

women_kurta

📂 Dataset

Dataset source:

👉 https://indofashion.github.io/

The original dataset contains ~106,000 images.

To make training manageable and balanced, a subset was created.

Dataset Subset
Classes	Images per class	Total
15	500	7,500
📊 Dataset Split
Split	Images
Train	5,250
Validation	1,125
Test	1,125
Total	7,500

Per class:

Train → 350

Validation → 75

Test → 75

🏗 Project Structure
indian-ethnic-clothing-classification/

configs/
data/
│
├── raw
│   └── indofashion_dataset
│
├── interim
│
└── processed
    └── dataset_subset
        ├── train
        ├── val
        └── test

docs/
notebooks/

outputs/
│
├── checkpoints
├── figures
│   ├── confusion_matrices
│   ├── model_comparison
│   └── training_curves
│
├── logs
└── reports

scripts/

src/
├── data
├── models
├── training
├── evaluation
└── utils

This structure separates:

data preparation

model training

evaluation

experiment outputs

and follows production-style machine learning project organization.

⚙️ Environment Setup
Python
Python 3.10
GPU
NVIDIA RTX 4060 Laptop GPU
CUDA
CUDA 12.8
PyTorch
torch 2.10.0 + cu128
📦 Installation

Clone the repository:

git clone https://github.com/your-username/indian-ethnic-clothing-classification.git

cd indian-ethnic-clothing-classification

Create environment:

conda create -n fashion-classifier python=3.10

conda activate fashion-classifier

Install dependencies:

pip install -r requirements.txt
🧹 Dataset Preparation

Prepare the dataset subset using:

python scripts/prepare_dataset.py

This script:

reads dataset metadata

groups images by class

samples 500 images per class

creates train / val / test splits

generates dataset summary

Output:

data/processed/dataset_subset
🔄 Data Augmentation

Training augmentations:

RandomResizedCrop

RandomHorizontalFlip

RandomRotation

RandomAffine

ColorJitter

Normalize

Validation / Test transforms:

Resize

Normalize

🧠 Models Used

The following pretrained CNN architectures were evaluated:

Model	Pretrained
ResNet50	ImageNet
MobileNetV2	ImageNet
EfficientNetB0	ImageNet
DenseNet121	ImageNet

The original classifier layer was replaced with a custom classification head.

Custom Classifier
Linear
ReLU
Dropout
Linear
🔬 Training Strategy

Two transfer learning strategies were used.

Frozen Backbone

Only classifier head is trained.

Used for:

ResNet50

EfficientNetB0

Benefits:

faster training

reduced overfitting

Full Fine-Tuning

Entire network is trained.

Used for:

MobileNetV2

DenseNet121

🏋️ Training Configuration

Loss Function

CrossEntropyLoss

Optimizer

AdamW

Scheduler

StepLR

Additional features:

GPU training

mixed precision training (AMP)

checkpoint saving

training logs

📈 Training Visualizations

Training curves are automatically generated.

Saved to:

outputs/figures/training_curves

Includes:

loss vs epoch

accuracy vs epoch

🧪 Model Evaluation

Evaluation metrics used:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

Generated using:

sklearn.metrics

Evaluation outputs:

classification_report.txt
confusion_matrix.png
test_metrics.json
🔎 Confusion Matrix Analysis

Confusion matrices visualize class-wise prediction errors.

Example confusions observed:

Class A	Confused With
leggings_and_salwars	dhoti_pants
gowns	women_kurta
dupattas	gowns

These occur due to visual similarity between clothing styles.

📊 Model Performance
Model	Test Accuracy	Macro F1	Weighted F1
MobileNetV2	77.42%	0.769	0.769
DenseNet121	77.24%	0.766	0.766
ResNet50	73.96%	0.735	0.735
EfficientNetB0	69.51%	0.685	0.685
🏆 Best Model

MobileNetV2

Reasons:

lightweight architecture

better generalization

fewer parameters

lower overfitting

📊 Outputs Generated
outputs/

checkpoints/
    resnet50_best.pth
    mobilenet_v2_best.pth
    efficientnet_b0_best.pth
    densenet121_best.pth

logs/

reports/
    classification_reports
    metrics_json
    model_comparison_summary.csv

figures/
    confusion_matrices
    training_curves
    model_comparison
🚀 Pipeline Features

The project implements a complete deep learning pipeline:

✔ dataset preparation
✔ balanced sampling
✔ PyTorch dataloaders
✔ data augmentation
✔ transfer learning models
✔ GPU training
✔ mixed precision training
✔ training visualization
✔ evaluation metrics
✔ confusion matrices
✔ automated multi-model evaluation
✔ model comparison plots

🔮 Future Improvements

Possible extensions:

training on full dataset

higher resolution input images

hyperparameter tuning

Vision Transformer models

deployment using FastAPI

model inference API

⚠️ Dataset Notice

The dataset is not included in this repository due to size limitations.

Download from:

https://indofashion.github.io/