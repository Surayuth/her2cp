# HER2CP: Conformal Prediction for HER2 Status Classification

This repository contains the implementation code for the paper Conformal Prediction for Uncertainty Quantification and Reliable HER2 Status Classification in Breast Cancer IHC Images.

## About

This project implements a Conformal Prediction (CP) framework for HER2 status classification in breast cancer IHC images. The framework:
- Quantifies uncertainty in HER2 status predictions
- Identifies borderline cases requiring additional testing
- Provides prediction sets with controlled error rates
- Uses handcrafted features with tree-based classifiers

## Prerequisites

- Python installed on your system
- At least 10GB free disk space (dataset: 6GB + extracted features)

## Environment Setup

1. Clone repository:
```bash
git clone https://github.com/Surayuth/her2cp.git
cd her2cp
```

2. Create and activate Python virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate # For Unix/macOS
# Or for Windows:
# .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Download dataset (6GB):
- Download [dataset file](https://drive.google.com/file/d/1KeKptikuHuCNVXRhfJueBlj67LJVbcIw/view?usp=sharing)
- Save to the project root directory

2. Extract dataset:
```bash
unzip Data_Chula.zip
```

3. Extract features (estimated time: 1-2 minutes):
```bash
python extract_feature.py --workers 8 --src ./Data_Chula --dst ./extracted_features
```

## Model Training

Example training command (using Decision Tree, completes in ~15 minutes):
```bash
python train.py --path extracted_features/feat_level_16_scale_0.25.csv --model dt
```

Available models: Decision Tree (dt), Random Forest (rf), Gradient Boosting Tree (gbt), and Extreme Gradient Boosting Tree (xgb).
Note: Training time varies by model complexity.

## Analysis & Results

The following notebooks reproduce the key results from the paper:

1. Miscoverage Rate Analysis: `Analysis/miscoverate.ipynb`
    - Verifies the framework's coverage guarantees at different significance levels

2. Ambiguity Rate Analysis: `Analysis/ambiguity.ipynb`
    - Analyzes the trade-off between prediction certainty and ambiguous cases

3. Accuracy Analysis: `Analysis/accuracy.ipynb`
    - Evaluates classification performance for positive and negative HER2 statuses

### Note
- All figures are standardized with y-axis limits set to [0, 100]
- The main differences between notebooks are in the aggregate functions used in the 4th cell





