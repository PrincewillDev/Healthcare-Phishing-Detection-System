# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ML system that classifies emails as phishing or legitimate, specialized for healthcare-context
phishing. Ships as a web app (FastAPI backend + minimal HTML/JS frontend), not a live mail server
integration. Built as a master's project deliverable.

**Status: greenfield.** Only `.gitignore`, `README.md`, and agent config exist so far. The
directory layout, setup commands, and model design below describe the intended target that future
work should build toward; treat them as the plan of record unless the user redirects.

## Tech stack

- Python 3.10+
- scikit-learn, xgboost, lightgbm — base models
- nltk, beautifulsoup4, tldextract — email/URL/domain preprocessing
- FastAPI — backend API
- Plain HTML/JS frontend, kept minimal (not a design priority)

## Setup

```
python -m venv venv
venv\Scripts\activate          # Windows (this environment); use source venv/bin/activate on POSIX
pip install -r requirements.txt
```

## Intended layout

- `data/raw/` — untouched source datasets, never edited in place, never committed (see data rules)
- `data/processed/` — cleaned, feature-extracted data with train/val/test splits
- `src/preprocessing/` — email parser, NLP cleaning, URL/domain extraction
- `src/features/` — feature engineering modules
- `src/models/` — training scripts; `src/models/artifacts/` holds saved models + metrics
- `src/api/` — FastAPI app
- `notebooks/` — exploration only, never imported by production code

## Model architecture

- Base models: Random Forest, XGBoost, LightGBM, each trained and evaluated independently
- Final model: stacking ensemble — a meta-classifier over the base models' outputs
- Performance targets: under 0.5% false positive rate, sub-second inference

The `train-baseline-models` skill covers training/retraining a single base model (fit on train,
evaluate on val with precision/recall/F1/FPR/AUC-ROC + confusion matrix, save artifact and a
matching `.json` metrics file to `src/models/artifacts/`). It deliberately does not touch ensemble
logic, feature engineering, or API wiring — those are separate steps.

## Hard constraints

Data handling (enforced by `.claude/rules/data-handling.md` for `data/**` and
`src/preprocessing/**`):

- Never commit raw dataset files (`data/raw/**`); keep them gitignored.
- Any healthcare email sample adapted or synthesized from a public dataset must be tagged as
  synthetic in its metadata. Never present adapted samples as real healthcare breach data in code,
  comments, or output.
- No external API calls (WHOIS, threat intel, live domain lookups) in the inference path. Use
  cached or offline data only, to keep latency predictable.

Conventions:

- No em dashes in any generated docs, comments, or writeups.
- Every feature extraction function must be unit-testable in isolation.
- When a trained model artifact is committed, its evaluation metrics go in the same commit — not
  the model file alone.
