import os
import pickle
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf

from src.api.schemas import (
    PredictRequest, PredictResponse,
    SearchRequest, SearchResponse, SearchResult,
)
from src.features.preprocess import TextNormalizer
from src.features.vectorize import vectorize_data

# ── Global state loaded at startup ────────────────────────────────────────────
state: dict = {}

DATASET = os.getenv("DATASET", "amazon")
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))


def _load_artifacts():
    """Load model, vectorizer, search index, and preprocessor config."""
    model_path = MODELS_DIR / f"{DATASET}_sentiment_model.pkl"
    vectorizer_path = MODELS_DIR / f"{DATASET}_vectorizer.pkl"
    search_index_path = MODELS_DIR / f"{DATASET}_search_index.pkl"

    for p in [model_path, vectorizer_path]:
        if not p.exists():
            raise RuntimeError(
                f"Artifact not found: {p}. "
                f"Run src/pipelines/train_pipeline.py first."
            )

    with open(model_path, "rb") as f:
        state["model"] = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        state["vectorizer"] = pickle.load(f)

    # Load search index if available (optional)
    if search_index_path.exists():
        with open(search_index_path, "rb") as f:
            state["searcher"] = pickle.load(f)
    else:
        state["searcher"] = None

    # Load matching preprocess config
    conf_path = Path("conf") / "preprocess" / f"{DATASET}.yaml"
    if not conf_path.exists():
        conf_path = Path("conf") / "preprocess" / "default.yaml"
    preprocess_cfg = OmegaConf.load(conf_path)
    state["normalizer"] = TextNormalizer(preprocess_cfg)

    # Load vectorizer config so we can reuse vectorize_data()
    vec_conf_path = Path("conf") / "vectorizer" / "tfidf.yaml"
    state["vec_cfg"] = OmegaConf.load(vec_conf_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    print(f"API ready — dataset='{DATASET}', models loaded from '{MODELS_DIR}'")
    yield
    state.clear()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NLP Sentiment API",
    description=(
        "Sentiment classifier and BM25 search engine "
        "for Amazon Fine Food Reviews and Sentiment140."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Status"])
def health():
    return {
        "status": "ok",
        "dataset": DATASET,
        "search_index_loaded": state.get("searcher") is not None,
    }


# ── /predict ──────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse, tags=["Classification"])
def predict(request: PredictRequest):
    """Classify a piece of text as positive (1) or negative (0)."""
    normalizer: TextNormalizer = state["normalizer"]
    model = state["model"]
    trained_vectorizer = state["vectorizer"]
    vec_cfg = state["vec_cfg"]

    cleaned = normalizer.transform([request.text])
    X = vectorize_data(cleaned, vec_cfg, trained_vectorizer=trained_vectorizer)

    label = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    confidence = float(np.max(proba))

    return PredictResponse(
        label=label,
        sentiment="positive" if label == 1 else "negative",
        confidence=round(confidence, 4),
    )


# ── /search ───────────────────────────────────────────────────────────────────
@app.post("/search", response_model=SearchResponse, tags=["Search"])
def search(request: SearchRequest):
    """Search the indexed corpus using BM25 with optional sentiment filtering."""
    searcher = state.get("searcher")
    normalizer: TextNormalizer = state["normalizer"]

    if searcher is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Search index not available. "
                "Run src/pipelines/search_pipeline.py to build it."
            ),
        )

    cleaned_query = normalizer._clean_single_text(request.query)
    raw_results = searcher.search(
        cleaned_query,
        n=request.n,
        sentiment_filter=request.sentiment_filter,
    )

    results = [
        SearchResult(
            document=r["document"],
            index=r["index"],
            score=r["score"],
            label=r.get("label"),
        )
        for r in raw_results
    ]

    return SearchResponse(
        query=request.query,
        results=results,
        total=len(results),
    )
