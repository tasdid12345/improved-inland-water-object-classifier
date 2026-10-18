# Calibrated LightGBM vs MLP for Outdoor Scene Region Classification

Detecting objects on inland waters from imagery is a hard, low-data task, and the motivating paper, "A Deep Network for Object Detection on Inland Waters" (2026), proposes a deep network for it. Its model is not described in enough detail to reproduce here, so this experiment uses a public proxy: outdoor image-region classification. It asks whether a simpler, class-balanced, calibrated gradient-boosting model can match or beat a standard neural-network baseline on that proxy.

**Question:** Can a class-balanced, calibrated LightGBM model beat a standard MLP neural-network baseline at classifying outdoor scene regions on the public UCI Image Segmentation dataset, as a stand-in for detecting objects on inland waters?

**Dataset:** UCI Image Segmentation (2,310 regions from seven outdoor images, 19 pixel-derived features), loaded with the `ucimlrepo` package. This is a public stand-in dataset, not the paper's own data. A stratified 80/20 split uses a fixed seed.

**Baseline:** a standard scikit-learn MLPClassifier on standardized features, with hidden size and regularization chosen by 5-fold CV on the training set.

**Improved model:** a LightGBM classifier with balanced class weights, with num_leaves and learning rate chosen by 5-fold CV on the training set. Its probabilities are calibrated with sigmoid (Platt) calibration.

Both models use the same preprocessing, split, search budget, and evaluation. The main metric is test macro-F1; accuracy and log-loss are secondary.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (charts in `figures/`).

## Reference

This project is motivated by the following paper:

> D Griesser, B Goldluecke, MO Franz (2026). *A Deep Network for Object Detection on Inland Waters*. 2026 IEEE/CVF Winter. https://ieeexplore.ieee.org/abstract/document/11492491/

Paper author: Prof. Bastian Goldlücke, Universität Konstanz (bastian.goldluecke@uni-konstanz.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
