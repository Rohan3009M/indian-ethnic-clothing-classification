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

This project builds a deep learning image classification system that recognizes Indian ethnic clothing categories using the IndoFashion dataset and transfer learning with PyTorch.

The goal is to build a complete end-to-end ML pipeline including:

Dataset preparation

Data augmentation

Transfer learning with pretrained CNNs

GPU-based training

Model evaluation

Confusion matrix analysis

Automated model comparison

📌 Project Overview

Indian ethnic fashion contains many garments that are visually similar, making classification challenging.

This project focuses on:

Applying transfer learning on pretrained CNN models

Comparing multiple architectures

Building a reproducible ML pipeline

Analyzing classification errors using confusion matrices

🧵 Clothing Categories

The model classifies 15 types of Indian ethnic clothing:

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

Dataset Source:
https://indofashion.github.io/

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

Per class distribution:

Train: 350

Validation: 75

Test: 75

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

Data preparation

Model training

Evaluation

Experiment outputs

and follows a production-style ML project organization.

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
Clone the repository
git clone https://github.com/Rohan3009M/indian-ethnic-clothing-classification.git

cd indian-ethnic-clothing-classification
Create environment
conda create -n fashion-classifier python=3.10

conda activate fashion-classifier
Install dependencies
pip install -r requirements.txt
🧹 Dataset Preparation

Prepare the dataset subset using:

python scripts/prepare_dataset.py

This script:

Reads dataset metadata

Groups images by class

Samples 500 images per class

Creates train / validation / test splits

Generates dataset summary

Output directory:

data/processed/dataset_subset
🔄 Data Augmentation
Training Transformations

RandomResizedCrop

RandomHorizontalFlip

RandomRotation

RandomAffine

ColorJitter

Normalize

Validation / Test Transformations

Resize

Normalize

🧠 Models Used

The following pretrained CNN architectures were evaluated:

Model	Pretrained
ResNet50	ImageNet
MobileNetV2	ImageNet
EfficientNetB0	ImageNet
DenseNet121	ImageNet

The original classifier was replaced with a custom classification head:

Linear
ReLU
Dropout
Linear
🔬 Training Strategy

Two transfer learning approaches were used.

Frozen Backbone

Only the classifier head is trained.

Used for:

ResNet50

EfficientNetB0

Benefits:

Faster training

Reduced overfitting

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

Additional training features:

GPU training

Mixed precision training (AMP)

Checkpoint saving

Training logs

📈 Training Visualizations

Training curves are automatically generated and saved to:

outputs/figures/training_curves

Includes:

Loss vs Epoch

Accuracy vs Epoch

🧪 Model Evaluation

Evaluation metrics:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

Generated using:

sklearn.metrics

Evaluation outputs include:

classification_report.txt

confusion_matrix.png

test_metrics.json

🔎 Confusion Matrix Analysis

Confusion matrices help visualize class-wise prediction errors.

Example misclassifications:

Class	Confused With
leggings_and_salwars	dhoti_pants
gowns	women_kurta
dupattas	gowns

These errors occur due to visual similarity between garments.

📊 Model Performance
Model	Test Accuracy	Macro F1	Weighted F1
MobileNetV2	77.42%	0.769	0.769
DenseNet121	77.24%	0.766	0.766
ResNet50	73.96%	0.735	0.735
EfficientNetB0	69.51%	0.685	0.685
🏆 Best Model

MobileNetV2

Reasons:

Lightweight architecture

Better generalization

Fewer parameters

Lower overfitting

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

This project implements a complete deep learning pipeline:

✔ Dataset preparation

✔ Balanced sampling

✔ PyTorch dataloaders

✔ Data augmentation

✔ Transfer learning models

✔ GPU training

✔ Mixed precision training

✔ Training visualization

✔ Evaluation metrics

✔ Confusion matrices

✔ Automated multi-model evaluation

✔ Model comparison plots

🔮 Future Improvements

Possible extensions:

Training on the full dataset

Higher resolution input images

Hyperparameter tuning

Vision Transformer models

Deployment using FastAPI

Building a model inference API

⚠️ Dataset Notice

The dataset is not included in this repository due to size limitations.

Download it from:

https://indofashion.github.io/