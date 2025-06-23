# Quant

# A-share Factor Sorting & CNN Stock Image Classifier

**Language notice:** All in-file comments remain in Chinese to preserve the original research context; English section headers have been added for reviewers.

## Project Overview
This repo contains two independent projects:

1. **FactorSort/** – Python scripts for A-share cross-sectional factor research  
   • Winsorization, decile sorting, Newey-West statistics  
   • Generates interactive plots of cumulative returns and factor descriptors

2. **CNN_Classifier/** – PyTorch pipeline for classifying candlestick chart images  
   • Custom DataLoader, training loop, and metric logger (`utils/logger.py`)  
   • Achieved 57 % directional accuracy vs. 50 % baseline

## Quick Start
```bash
pip install -r requirements.txt
python FactorSort/main.py      # reproduces Figure 3 in the pape
python CNN_Classifier/train.py --epochs 20
