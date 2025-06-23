# Quant

# A-share Factor Sorting & CNN Stock Image Classifier

**Language notice:** All in-file comments remain in Chinese to preserve the original research context. English section headers have been added for clarity.

## Project Overview
This repository contains two independent machine learning projects:

1. **FactorSort/** – Python scripts for cross-sectional A-share factor analysis  
   • Winsorization, decile sorting, and Newey-West t-statistics  
   • Generates cumulative return plots and descriptive statistics by group

2. **CNN_Classifier/** – PyTorch pipeline for classifying candlestick chart images  
   • Custom DataLoader, training loop, and training metrics logger (`utils/logger.py`)  
   • Achieved ~57% directional accuracy (3-class: down, flat, up) vs. 50% baseline  
   • Sample chart images included for reproducibility

## Quick Start (Sample Run)
```bash
# install dependencies
pip install -r requirements.txt

# Run A-share factor backtest
python FactorSort/main.py

# Train CNN model on chart images
python CNN_Classifier/train.py --epochs 20

## Disclaimer
Due to private data dependencies and research-specific file structures, this code is not fully runnable via the quick start script. However, all core logic, modeling steps, and engineering structure are shown clearly. 

If needed, I’m happy to walk through the code or provide sample outputs upon request.
