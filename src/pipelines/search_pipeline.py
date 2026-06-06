import pickle
from pathlib import Path

import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.data.ingest import load_data
from src.features.preprocess import TextNormalizer
from src.features.vectorize import vectorize_data
from src.search.bm25 import BM25Searcher


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def build_search_index(cfg: DictConfig):
    """
    Load dataset, run predictions with saved model, build a
    sentiment-aware BM25 index, and serialize it to disk.
    """
    dataset_name = cfg.dataset.name
    print(f"Building search index for: {dataset_name.upper()}")

    # 1. Load saved model + vectorizer
    model_path = Path(f"models/{dataset_name}_sentiment_model.pkl")
    vectorizer_path = Path(f"models/{dataset_name}_vectorizer.pkl")

    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            f"Trained artifacts not found in models/. "
            f"Run train_pipeline.py first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        trained_vectorizer = pickle.load(f)

    # 2. Load full dataset (no split needed — index everything)
    df = load_data(cfg)

    # 3. Preprocess text
    normalizer = TextNormalizer(cfg.preprocess)
    print("Preprocessing corpus for indexing...")
    cleaned = normalizer.transform(df["raw_text"])

    # 4. Predict sentiment for every document
    print("Predicting sentiment labels for corpus...")
    X = vectorize_data(cleaned, cfg.vectorizer, trained_vectorizer=trained_vectorizer)
    labels = model.predict(X).tolist()

    # 5. Build BM25 index with labels
    print(f"Indexing {len(cleaned):,} documents...")
    searcher = BM25Searcher(corpus=cleaned, labels=labels)

    # 6. Save index
    index_path = f"models/{dataset_name}_search_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump(searcher, f)

    print(f"Search index saved to: {index_path}")

    # Quick smoke-test
    print("\n--- Smoke Test ---")
    pos_results = searcher.search("great product amazing", n=3, sentiment_filter=1)
    neg_results = searcher.search("terrible awful waste", n=3, sentiment_filter=0)

    print(f"Positive results ({len(pos_results)}):")
    for r in pos_results:
        print(f"  [{r['score']:.3f}] {r['document'][:80]}")

    print(f"Negative results ({len(neg_results)}):")
    for r in neg_results:
        print(f"  [{r['score']:.3f}] {r['document'][:80]}")


if __name__ == "__main__":
    build_search_index()
