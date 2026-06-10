import os
import pickle
import sys
import io
from pathlib import Path

import streamlit as st
import numpy as np
import requests
from dotenv import load_dotenv

# Reconfigure standard output streams to UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except AttributeError:
        pass

load_dotenv()

# App Page Setup
st.set_page_config(
    page_title="NLP Intelligence Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #27ae60;
    }
    .report-card {
        padding: 1.5rem;
        border-radius: 8px;
        background-color: #1f2937;
        margin-bottom: 1rem;
        border-left: 5px solid #2ecc71;
    }
    </style>
""", unsafe_allow_html=True)

# ── Sidebar Config ────────────────────────────────────────────────────────────
st.sidebar.title("🧠 NLP Intelligence System")
st.sidebar.subheader("Configuration")

# Dataset Selection
dataset = st.sidebar.selectbox(
    "Select Target Dataset",
    options=["amazon", "sentiment140"],
    format_func=lambda x: "Amazon Reviews (Food)" if x == "amazon" else "Sentiment140 (Twitter)"
)

# API Mode selection
api_url = st.sidebar.text_input("FastAPI Backend URL", value="http://localhost:8000")

# Check backend status
backend_live = False
try:
    resp = requests.get(f"{api_url}/health", timeout=1.5)
    if resp.status_code == 200:
        backend_live = True
        st.sidebar.success("🟢 FastAPI Backend Live!")
    else:
        st.sidebar.warning("🟡 API returned warning. Running locally.")
except Exception:
    st.sidebar.warning("🔴 API Offline. Running in Local Mode.")

# ── Local Mode Loader ─────────────────────────────────────────────────────────
@st.cache_resource
def load_local_model_artifacts(ds_name: str):
    """Loads preprocessors and models locally if API is offline."""
    from src.features.preprocess import TextNormalizer
    from src.features.vectorize import vectorize_data
    from omegaconf import OmegaConf

    models_dir = Path("models")
    model_path = models_dir / f"{ds_name}_sentiment_model.pkl"
    vectorizer_path = models_dir / f"{ds_name}_vectorizer.pkl"
    search_path = models_dir / f"{ds_name}_search_index.pkl"
    w2v_path = models_dir / f"{ds_name}_word2vec.model"

    artifacts = {}
    
    # Load Preprocessor Config
    conf_path = Path("conf") / "preprocess" / f"{ds_name}.yaml"
    if not conf_path.exists():
        conf_path = Path("conf") / "preprocess" / "default.yaml"
    
    preprocess_cfg = OmegaConf.load(conf_path)
    artifacts["normalizer"] = TextNormalizer(preprocess_cfg)

    # Load Model & Vectorizer
    if model_path.exists() and vectorizer_path.exists():
        with open(model_path, "rb") as f:
            artifacts["model"] = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            artifacts["vectorizer"] = pickle.load(f)
    else:
        st.error(f"Trained model artifacts not found locally for '{ds_name}'. Run train_pipeline.py first!")

    # Load searcher
    if search_path.exists():
        with open(search_path, "rb") as f:
            artifacts["searcher"] = pickle.load(f)
    else:
        artifacts["searcher"] = None

    # Load Word2Vec
    if w2v_path.exists():
        from src.features.embeddings import load_embeddings
        artifacts["w2v"] = load_embeddings(str(w2v_path))
    else:
        artifacts["w2v"] = None

    return artifacts

# Load local models if backend is offline
local_artifacts = None
if not backend_live:
    local_artifacts = load_local_model_artifacts(dataset)

# ── Tabs Setup ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🎯 Sentiment Classification",
    "🔍 Semantic Search Engine",
    "📊 Word Embeddings Explorer"
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: Sentiment Classifier
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("🎯 Real-Time Sentiment Classification")
    st.write("Input raw text below to classify its sentiment as Positive or Negative.")

    text_input = st.text_area(
        "Enter text to analyze:",
        placeholder="Type here (e.g., 'The delivery was quick and the food tasted absolutely amazing!')",
        height=120
    )

    if st.button("Classify Sentiment", key="classify_btn"):
        if not text_input.strip():
            st.warning("Please enter some text first!")
        else:
            if backend_live:
                # API Call
                try:
                    res = requests.post(f"{api_url}/predict", json={"text": text_input}).json()
                    label = res["label"]
                    sentiment = res["sentiment"]
                    confidence = res["confidence"]
                except Exception as e:
                    st.error(f"Error calling backend: {e}")
                    label, sentiment, confidence = None, None, None
            else:
                # Local Call
                if "model" in local_artifacts:
                    normalizer = local_artifacts["normalizer"]
                    model = local_artifacts["model"]
                    vectorizer = local_artifacts["vectorizer"]
                    
                    from src.features.vectorize import vectorize_data
                    from omegaconf import OmegaConf
                    
                    cleaned = normalizer.transform([text_input])
                    vec_cfg = OmegaConf.create({"method": "tfidf"})
                    X = vectorize_data(cleaned, vec_cfg, trained_vectorizer=vectorizer)
                    
                    label = int(model.predict(X)[0])
                    proba = model.predict_proba(X)[0]
                    confidence = float(np.max(proba))
                    sentiment = "positive" if label == 1 else "negative"
                else:
                    label = None

            if label is not None:
                # Display Results
                st.subheader("Result")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if label == 1:
                        st.markdown(
                            "<div style='background-color:#1e3a20; padding:20px; border-radius:10px; border-left: 6px solid #2ecc71;'>"
                            "<h3 style='margin:0; color:#2ecc71;'>Positive Sentiment Verdict</h3>"
                            f"</div>", 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            "<div style='background-color:#3b1e1e; padding:20px; border-radius:10px; border-left: 6px solid #e74c3c;'>"
                            "<h3 style='margin:0; color:#e74c3c;'>Negative Sentiment Verdict</h3>"
                            f"</div>", 
                            unsafe_allow_html=True
                        )
                
                with col2:
                    st.metric(label="Confidence Level", value=f"{confidence * 100:.2f}%")
                    st.progress(confidence)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: Semantic Search Engine
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("🔍 BM25 Search Engine with Sentiment Filtering")
    st.write("Search indexed corpus documents with optional sentiment overrides.")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        query_input = st.text_input("Enter search query:", placeholder="e.g. delicious cookies, bad service")
    with col2:
        k_results = st.slider("Number of results", min_value=1, max_value=20, value=5)
    with col3:
        filter_sentiment = st.selectbox(
            "Sentiment Filter",
            options=[None, 1, 0],
            format_func=lambda x: "All Sentiments" if x is None else ("Positive Only" if x == 1 else "Negative Only")
        )

    if st.button("Search Corpus", key="search_btn"):
        if not query_input.strip():
            st.warning("Please enter a query!")
        else:
            results = None
            if backend_live:
                try:
                    payload = {
                        "query": query_input,
                        "n": k_results,
                        "sentiment_filter": filter_sentiment
                    }
                    res = requests.post(f"{api_url}/search", json=payload).json()
                    results = res.get("results", [])
                except Exception as e:
                    st.error(f"API Search failed: {e}")
            else:
                # Local Search
                searcher = local_artifacts.get("searcher") if local_artifacts else None
                if searcher:
                    normalizer = local_artifacts["normalizer"]
                    cleaned_q = normalizer._clean_single_text(query_input)
                    results = searcher.search(cleaned_q, n=k_results, sentiment_filter=filter_sentiment)
                else:
                    st.info("Local BM25 search index file not found. Run search_pipeline.py to index your corpus!")

            if results:
                st.subheader(f"Top Results ({len(results)} matches)")
                for i, r in enumerate(results):
                    lbl_badge = "🟢 Positive" if r.get("label") == 1 else ("🔴 Negative" if r.get("label") == 0 else "")
                    st.markdown(
                        f"<div class='report-card' style='border-left-color: {'#2ecc71' if r.get('label') == 1 else '#e74c3c'};'>"
                        f"<strong>Match #{i+1}</strong> (Score: {r['score']:.4f})  |  {lbl_badge}<br/>"
                        f"<p style='margin-top:10px; font-style:italic;'>\"{r['document']}\"</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            elif results == []:
                st.info("No matching documents found.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: Word Embeddings Explorer
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("📊 Word Embeddings & Semantic Association")
    st.write("Lookup semantic similarities using static trained dense vectors.")

    w2v_model = None
    if backend_live:
        st.info("Embedding lookup currently runs in Local Mode. Ensure Word2Vec artifacts are built.")
    
    # Check local artifacts
    local_art = load_local_model_artifacts(dataset)
    w2v_model = local_art.get("w2v")

    if w2v_model is None:
        st.warning("Word2Vec embedding model not found. Run the notebooks/embeddings_viz.ipynb notebook to train it!")
    else:
        st.success("Word2Vec Model Loaded Successfully.")
        
        lookup_word = st.text_input("Enter a seed word to find similar concepts:", value="delicious").lower().strip()
        
        if st.button("Find Similar Words"):
            if lookup_word not in w2v_model.wv:
                st.error(f"'{lookup_word}' is out of vocabulary!")
            else:
                from src.features.embeddings import most_similar
                sims = most_similar(w2v_model, lookup_word, n=10)
                
                # Plot/Table
                import pandas as pd
                df_sims = pd.DataFrame(sims, columns=["Word", "Cosine Similarity"])
                
                col_left, col_right = st.columns([1, 1])
                with col_left:
                    st.dataframe(df_sims.style.background_gradient(cmap="Greens", subset=["Cosine Similarity"]))
                
                with col_right:
                    # Render a simple horizontal bar chart
                    st.bar_chart(df_sims.set_index("Word"))
