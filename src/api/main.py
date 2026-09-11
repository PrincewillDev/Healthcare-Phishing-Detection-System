"""FastAPI classification endpoint for the phishing detector.

Loads the already-fitted artifacts (TF-IDF vectorizer, feature scaler, three
base models, stacking meta-classifier) once at startup and reuses them for
every request. The inference pipeline reproduces, in order, the exact
processing used during training:

  1. Clean the text (src/preprocessing/clean_dataset.py's encoding/HTML
     normalization -- decode MIME words, fix mojibake, strip HTML tags)
  2. Extract text/lexical features (src/features/text_features.py) and
     transform them through the already-fitted TF-IDF vectorizer
  3. Extract URL/domain features (src/features/url_features.py)
  4. Scale the 12 dense features (6 text/lexical + 6 URL) with the
     already-fitted StandardScaler -- never refit
  5. hstack sparse TF-IDF (5000 dims) with the scaled dense block
     (12 dims) into the same 5,012-dim feature vector used in training
  6. Run the three base models, feed their probabilities into the
     stacking meta-classifier, threshold the ensemble probability

Each response also carries a SHAP-based "explanation": the top 10
features (by absolute SHAP value) that drove each base model's
prediction, with the direction each one pushed (toward phishing or
toward legitimate). See compute_explanation() for why this is reported
per base model rather than combined into one ranking.

No retraining, refitting, or external network calls happen anywhere in
this module.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from email import message_from_bytes
from email.message import Message
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import shap
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "src" / "models" / "artifacts"

sys.path.insert(0, str(ROOT / "src" / "features"))
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))

from text_features import extract_row_features  # noqa: E402
from url_features import extract_row_url_features  # noqa: E402
from clean_dataset import clean_text_encoding_and_html  # noqa: E402

logger = logging.getLogger("phishing_api")
logging.basicConfig(level=logging.INFO)

BASE_MODEL_NAMES = ["random_forest", "xgboost", "lightgbm"]

TEXT_DENSE_COLS = [
    "urgency_score", "healthcare_term_count", "word_count",
    "avg_word_length", "exclamation_count", "capitalized_word_count",
]
URL_DENSE_COLS = [
    "url_count", "has_ip_literal", "has_url_shortener",
    "avg_domain_entropy", "suspicious_tld", "has_at_symbol",
]
DENSE_COLS = TEXT_DENSE_COLS + URL_DENSE_COLS

DEFAULT_THRESHOLD = 0.67

TOP_N_EXPLANATION_FEATURES = 10

# Background sample for RandomForest's SHAP explainer (see build_shap_explainers
# for why RF needs one and XGBoost/LightGBM don't). Precomputed once from train
# (20 rows, random_state=42, see src/models/artifacts/shap_background_sample.npz)
# so the API never needs the full train_final.npz at startup.
SHAP_BACKGROUND_PATH = ARTIFACTS_DIR / "shap_background_sample.npz"

artifacts: dict = {}


def load_artifacts() -> dict:
    loaded = {
        "tfidf_vectorizer": joblib.load(ARTIFACTS_DIR / "tfidf_vectorizer.pkl"),
        "feature_scaler": joblib.load(ARTIFACTS_DIR / "feature_scaler.pkl"),
        "random_forest": joblib.load(ARTIFACTS_DIR / "random_forest.pkl"),
        "xgboost": joblib.load(ARTIFACTS_DIR / "xgboost.pkl"),
        "lightgbm": joblib.load(ARTIFACTS_DIR / "lightgbm.pkl"),
        "stacking_meta": joblib.load(ARTIFACTS_DIR / "stacking_meta.pkl"),
    }
    loaded["feature_names"] = (
        list(loaded["tfidf_vectorizer"].get_feature_names_out()) + DENSE_COLS
    )
    loaded.update(build_shap_explainers(loaded))
    return loaded


def build_shap_explainers(loaded: dict) -> dict:
    """TreeExplainer for each base model, loaded once at startup.

    XGBoost and LightGBM use the default "tree_path_dependent" algorithm
    (fast, no background data needed) -- verified correct by checking that
    each SHAP row's sum plus the expected value reproduces the model's raw
    margin output.

    sklearn's RandomForestClassifier hits a known numerical bug in that same
    fast path at this feature width (5,012 columns): the additivity check
    fails by many orders of magnitude (sum of SHAP values off by ~1e33 from
    the actual prediction), which means the fast-path values for RF are
    outright wrong here, not just imprecise. Switching RF to
    feature_perturbation="interventional" with model_output="probability"
    and a small real background sample avoids that code path entirely and
    reproduces predict_proba almost exactly (verified: sum of SHAP values +
    expected_value landed within 0.0003 of the true probability across
    several rows). The background sample only needs to be small (20 rows
    keeps this under ~150ms) since it is just a reference point for the
    interventional algorithm, not training data.
    """
    background = np.load(SHAP_BACKGROUND_PATH)["background"]

    explainer_rf = shap.TreeExplainer(
        loaded["random_forest"],
        data=background,
        feature_perturbation="interventional",
        model_output="probability",
    )
    explainer_xgb = shap.TreeExplainer(loaded["xgboost"])
    explainer_lgbm = shap.TreeExplainer(loaded["lightgbm"])

    return {
        "shap_explainer_random_forest": explainer_rf,
        "shap_explainer_xgboost": explainer_xgb,
        "shap_explainer_lightgbm": explainer_lgbm,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifacts.update(load_artifacts())
    logger.info("Loaded all artifacts: %s", ", ".join(artifacts.keys()))
    yield
    artifacts.clear()


app = FastAPI(title="Healthcare Phishing Detection API", lifespan=lifespan)


class ClassifyRequest(BaseModel):
    email_text: str
    threshold: Optional[float] = Field(default=None)


class BaseModelScores(BaseModel):
    random_forest: float
    xgboost: float
    lightgbm: float


class ExplanationFeature(BaseModel):
    feature: str
    shap_value: float
    direction: str


class ClassifyResponse(BaseModel):
    classification: str
    ensemble_probability: float
    threshold_used: float
    base_model_scores: BaseModelScores
    input_method: str
    explanation: dict[str, list[ExplanationFeature]]


def validate_threshold(threshold: Optional[float]) -> float:
    if threshold is None:
        return DEFAULT_THRESHOLD
    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(
            status_code=400,
            detail=f"threshold must be between 0 and 1, got {threshold}",
        )
    return threshold


def extract_eml_subject_body(raw_bytes: bytes) -> tuple[str, str]:
    """Mirrors merge_datasets._extract_nazario_message for consistency."""
    msg: Message = message_from_bytes(raw_bytes)
    subject = msg.get("Subject", "") or ""

    def decode_payload(part) -> str:
        try:
            payload = part.get_payload(decode=True)
        except Exception:
            return ""
        if not payload:
            return ""
        charset = part.get_content_charset() or "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            return payload.decode("utf-8", errors="replace")

    plain_parts, html_parts = [], []
    if msg.is_multipart():
        for part in msg.walk():
            if part.is_multipart():
                continue
            ctype = part.get_content_type()
            text = decode_payload(part)
            if not text:
                continue
            if ctype == "text/plain":
                plain_parts.append(text)
            elif ctype == "text/html":
                html_parts.append(text)
    else:
        text = decode_payload(msg)
        if msg.get_content_type() == "text/html":
            html_parts.append(text)
        else:
            plain_parts.append(text)

    if plain_parts:
        body = "\n".join(plain_parts)
    elif html_parts:
        from bs4 import BeautifulSoup
        body = "\n".join(
            BeautifulSoup(h, "html.parser").get_text(separator=" ", strip=True)
            for h in html_parts
        )
    else:
        body = ""

    return subject, body


def build_feature_vector(email_text: str) -> sparse.csr_matrix:
    cleaned_text = clean_text_encoding_and_html(email_text)

    tfidf_vec = artifacts["tfidf_vectorizer"].transform([cleaned_text])

    text_feats = extract_row_features(cleaned_text)
    url_feats = extract_row_url_features(cleaned_text)
    dense_row = [float(text_feats[c]) for c in TEXT_DENSE_COLS] + \
                [float(url_feats[c]) for c in URL_DENSE_COLS]
    dense_array = np.array([dense_row], dtype=float)

    scaled_dense = artifacts["feature_scaler"].transform(dense_array)
    dense_sparse = sparse.csr_matrix(scaled_dense)

    return sparse.hstack([tfidf_vec, dense_sparse], format="csr")


def _positive_class_shap_row(shap_output) -> np.ndarray:
    """Normalize a TreeExplainer.shap_values() result to a 1D array of
    per-feature contributions toward the positive (phishing) class, for a
    single input row. Handles both output conventions SHAP uses across
    model types/versions: a list of one array per class, or a single array
    (already positive-class margin, as XGBoost/LightGBM return) that may be
    2D (n_samples, n_features) or 3D (n_samples, n_features, n_classes).
    """
    if isinstance(shap_output, list):
        return np.asarray(shap_output[1])[0]
    arr = np.asarray(shap_output)
    if arr.ndim == 3:
        return arr[0, :, 1]
    return arr[0]


def top_features_from_shap(sv_row: np.ndarray, feature_names: list[str]) -> list[ExplanationFeature]:
    top_idx = np.argsort(-np.abs(sv_row))[:TOP_N_EXPLANATION_FEATURES]
    return [
        ExplanationFeature(
            feature=feature_names[i],
            shap_value=float(sv_row[i]),
            direction="phishing" if sv_row[i] > 0 else "legitimate",
        )
        for i in top_idx
    ]


def compute_explanation(x_dense: np.ndarray) -> dict[str, list[ExplanationFeature]]:
    """Top-10 SHAP features per base model, reported separately rather than
    combined into one ranking. RF's SHAP values are in probability units
    (see build_shap_explainers) while XGBoost/LightGBM's are in log-odds
    (margin) units -- averaging or summing across those scales directly
    would silently misrepresent which model's signal actually dominates, so
    each model's contribution list stands on its own instead.
    """
    feature_names = artifacts["feature_names"]

    sv_rf = artifacts["shap_explainer_random_forest"].shap_values(x_dense, check_additivity=False)
    sv_xgb = artifacts["shap_explainer_xgboost"].shap_values(x_dense, check_additivity=False)
    sv_lgbm = artifacts["shap_explainer_lightgbm"].shap_values(x_dense, check_additivity=False)

    return {
        "random_forest": top_features_from_shap(_positive_class_shap_row(sv_rf), feature_names),
        "xgboost": top_features_from_shap(_positive_class_shap_row(sv_xgb), feature_names),
        "lightgbm": top_features_from_shap(_positive_class_shap_row(sv_lgbm), feature_names),
    }


def run_inference(email_text: str, threshold: float, input_method: str) -> ClassifyResponse:
    start = time.perf_counter()

    X = build_feature_vector(email_text)

    base_scores = {
        name: float(artifacts[name].predict_proba(X)[:, 1][0])
        for name in BASE_MODEL_NAMES
    }
    meta_input = np.array([[base_scores[name] for name in BASE_MODEL_NAMES]])
    ensemble_probability = float(
        artifacts["stacking_meta"].predict_proba(meta_input)[:, 1][0]
    )
    classification = "phishing" if ensemble_probability >= threshold else "legitimate"

    shap_start = time.perf_counter()
    x_dense = X.toarray()
    explanation = compute_explanation(x_dense)
    shap_elapsed = time.perf_counter() - shap_start

    elapsed = time.perf_counter() - start
    logger.info(
        "input_method=%s classification=%s ensemble_probability=%.4f "
        "threshold=%.4f response_time_ms=%.2f shap_time_ms=%.2f",
        input_method, classification, ensemble_probability, threshold,
        elapsed * 1000, shap_elapsed * 1000,
    )

    return ClassifyResponse(
        classification=classification,
        ensemble_probability=ensemble_probability,
        threshold_used=threshold,
        base_model_scores=BaseModelScores(**base_scores),
        input_method=input_method,
        explanation=explanation,
    )


@app.get("/health")
async def health() -> dict:
    expected = {
        "tfidf_vectorizer", "feature_scaler", "random_forest",
        "xgboost", "lightgbm", "stacking_meta",
    }
    loaded = expected.issubset(artifacts.keys())
    return {"status": "ok" if loaded else "error", "artifacts_loaded": loaded}


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: Request):
    """Accepts EITHER a JSON body ({"email_text": ..., "threshold": ...}) or
    a multipart/form-data upload (fields "file" = a .eml file, optional
    "threshold"). FastAPI cannot declare a JSON-body Pydantic model and
    File/Form parameters on the same path operation, since the two imply
    different request content-types, so content-type is dispatched on
    manually here.
    """
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "filename"):
            raise HTTPException(
                status_code=400,
                detail="Multipart upload must include a 'file' field",
            )
        if not upload.filename.lower().endswith(".eml"):
            raise HTTPException(
                status_code=400,
                detail=f"Only .eml file uploads are supported, got '{upload.filename}'",
            )

        raw_threshold = form.get("threshold")
        threshold = None
        if raw_threshold not in (None, ""):
            try:
                threshold = float(raw_threshold)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"threshold must be a number, got '{raw_threshold}'",
                )
        validated_threshold = validate_threshold(threshold)

        raw_bytes = await upload.read()
        subject, body = extract_eml_subject_body(raw_bytes)
        email_text = f"{subject}\n\n{body}".strip() if subject else body.strip()
        return run_inference(email_text, validated_threshold, "eml_upload")

    try:
        body_json = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Request body must be JSON with an 'email_text' field, "
                    "or a multipart/form-data upload with a 'file' field",
        )

    try:
        payload = ClassifyRequest(**body_json)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors())

    validated_threshold = validate_threshold(payload.threshold)
    return run_inference(payload.email_text, validated_threshold, "raw_text")


# Static demo frontend. Served from this same FastAPI process (rather than a
# separate static server) since the whole system is a single local demo
# deliverable, not a deployed product -- one process to start, no CORS setup
# needed. Mounted after the API routes so /classify and /health are matched
# first regardless of mount order.
STATIC_DIR = Path(__file__).resolve().parent / "static"
SAMPLES_DIR = ROOT / "test_samples"

app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
