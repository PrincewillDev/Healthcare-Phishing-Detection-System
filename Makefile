# Convenience wrapper around the project's existing scripts. Every target
# below is just a fixed sequence of the same commands documented in
# CLAUDE.md / README.md -- it does not add any new behavior.

ifeq ($(OS),Windows_NT)
PYTHON := venv/Scripts/python.exe
else
PYTHON := venv/bin/python
endif

.DEFAULT_GOAL := help

.PHONY: help setup data features train tune benchmark api all clean

help:
	@echo "Available targets:"
	@echo "  make setup      - create venv and install requirements.txt"
	@echo "  make data       - run the full data pipeline (download, merge, clean, sample/split)"
	@echo "  make features   - extract text + URL features and assemble the final feature matrix"
	@echo "  make train      - train the three base models and the stacking ensemble"
	@echo "  make tune       - run threshold tuning on the trained ensemble"
	@echo "  make benchmark  - evaluate existing trained models on the test set (no retraining)"
	@echo "  make api        - start the FastAPI server (uvicorn, --reload, port 8123)"
	@echo "  make all        - run data, features, train, and tune in sequence (full rebuild)"
	@echo "  make clean      - delete data/processed, data/raw, and generated model artifacts (destructive)"

setup:
	python -m venv venv
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) src/preprocessing/run_data_pipeline.py
	$(PYTHON) src/preprocessing/merge_datasets.py
	$(PYTHON) src/preprocessing/clean_dataset.py --apply-non-email-filter --write
	$(PYTHON) src/preprocessing/sample_and_split.py

features:
	$(PYTHON) src/features/build_text_features.py
	$(PYTHON) src/features/build_url_features.py
	$(PYTHON) src/features/assemble_final_features.py

train:
	$(PYTHON) src/models/train_random_forest.py
	$(PYTHON) src/models/train_xgboost.py
	$(PYTHON) src/models/train_lightgbm.py
	$(PYTHON) src/models/train_stacking_ensemble.py

tune:
	$(PYTHON) src/models/tune_threshold.py

benchmark:
	$(PYTHON) src/models/run_benchmark.py

api:
	$(PYTHON) -m uvicorn src.api.main:app --reload --port 8123

all: data features train tune

clean:
	@echo "WARNING: this will permanently delete:"
	@echo "  - data/processed/ (entire directory)"
	@echo "  - data/raw/ (entire directory)"
	@echo "  - generated model artifacts in src/models/artifacts/ (*.pkl, *.joblib)"
	@echo "data/synthetic/ and the tracked *.json metrics files are NOT touched."
	@read -p "Type 'yes' to continue: " confirm; \
	if [ "$$confirm" != "yes" ]; then \
		echo "Aborted."; \
		exit 1; \
	fi
	rm -rf data/processed data/raw
	rm -f src/models/artifacts/*.pkl src/models/artifacts/*.joblib
	@echo "Clean complete."
