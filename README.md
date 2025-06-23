# Quant

# A-share Factor Sorting & CNN Stock Image Classifier

**Language Notice:** All in-file comments are written in Chinese to preserve the original research context. English section headers have been added for clarity and accessibility.

## Project Overview

This repository showcases two independent machine learning projects combining financial modeling and deep learning:

1. **FactorSort/** – Python pipeline for cross-sectional factor analysis on the Chinese A-share market  
   • Implements winsorization, decile sorting, and Newey-West adjusted t-statistics  
   • Outputs cumulative return plots and factor group descriptors  

2. **CNN_Classifier/** – PyTorch pipeline for classifying stock candlestick chart images  
   • Contains core training script (`train.py`) demonstrating model architecture and training logic  
   • Achieves ~57% directional accuracy (3-class: down, flat, up) compared to 50% baseline  
   • Additional modules such as `DataLoader` and metrics logger will be uploaded soon  

## Quick Start (Sample Only)

```bash
# Install dependencies
pip install -r requirements.txt

# Run A-share factor analysis
python FactorSort/main.py

# Train CNN model on chart images
python CNN_Classifier/train.py --epochs 20
