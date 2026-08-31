---
name: train-baseline-models
description: Train and evaluate a baseline ML model (RF, XGBoost, or LightGBM) on the processed phishing dataset, and report standard classification metrics.
---

# Train Baseline Model

Use this when training or retraining any single baseline model.

## Steps
1. Load processed data from /data/processed (train/val/test splits)
2. Fit the specified model on the training set
3. Evaluate on the validation set: precision, recall, F1, false positive
   rate, AUC-ROC
4. Print a confusion matrix
5. Save the model artifact to /src/models/artifacts with a filename
   that includes the model type and date
6. Save the metrics alongside it as a .json file, same base filename

## Out of scope
- Do not touch the stacking ensemble logic here, that is a separate step
- Do not modify feature engineering code from this skill
- Do not deploy or wire the model into the API from this skill