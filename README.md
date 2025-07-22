# Quant

## 📈 A-share Factor Sorting & 📊 CNN Stock Image Classifier

> **Disclaimer:** Only partial source code is provided in this repository due to data licensing and confidentiality restrictions. The uploaded modules highlight the core logic and structure of the original research.

---

## 🧠 Project Overview

This repository presents two standalone machine learning pipelines combining **quantitative finance** and **deep learning**:

### 1. `FactorSort/`: Cross-Sectional Factor Analysis on China A-Shares
- Partial code for a Python pipeline analyzing equity alpha factors
- Implements:
  - Winsorization, z-scoring, decile sorting
  - Newey-West t-statistics for significance testing
- Outputs (in full version):
  - Cumulative returns
  - Factor performance reports
- 🔒 **Note:** Proprietary datasets and backtest results not included

### 2. `CNN_Classifier/`: Stock Chart Image Classification with PyTorch
- Selected core scripts demonstrating:
  - CNN model architecture and training pipeline (`train.py`)
  - Modularized training config and evaluation logic
- Achieves ~57% accuracy on 3-class directional prediction (`down`, `flat`, `up`)
- 🔜 Additional modules (e.g., `DataLoader`, logger) to be uploaded incrementally
- 🔒 **Note:** Image datasets and model checkpoints withheld due to confidentiality

---

## 🚀 Quick Start (Sample Only)

```bash
# Install required packages
pip install -r requirements.txt

# Run partial factor analysis pipeline
python FactorSort/main.py

# Train CNN on candlestick images (with your own data)
python CNN_Classifier/train.py --epochs 20
