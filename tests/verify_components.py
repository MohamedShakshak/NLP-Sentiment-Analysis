import pandas as pd
from omegaconf import OmegaConf

from src.features.preprocess import TextNormalizer
from src.features.vectorize import vectorize_data
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.search.bm25 import BM25Searcher


def test_components():
    print("--- Running Component Tests ---")

    # 1. Test TextNormalizer
    print("1. Testing TextNormalizer...")
    norm_cfg = OmegaConf.create({
        "lowercase": True,
        "remove_stopwords": True,
        "method": "lemmatization",
        "strip_html": True,
        "handle_social_tokens": False,
        "expand_contractions": False,
        "collapse_elongated": False,
    })
    normalizer = TextNormalizer(norm_cfg)
    raw_texts = pd.Series([
        "<html>Hello World! This is a test.</html>",
        "Another text."
    ])
    cleaned = normalizer.transform(raw_texts)
    print(f"   Input:  {list(raw_texts)}")
    print(f"   Output: {cleaned}")
    assert cleaned[0] == "hello world test", \
        f"Expected 'hello world test', got '{cleaned[0]}'"

    # 2. Test Vectorization
    print("2. Testing Vectorizer (TF-IDF)...")
    vec_cfg = OmegaConf.create({"method": "tfidf", "max_features": 10})
    X, vec = vectorize_data(cleaned, vec_cfg)
    print(f"   TF-IDF Matrix Shape: {X.shape}")
    assert X.shape[0] == 2, f"Expected 2 rows, got {X.shape[0]}"

    # 3. Test Dynamic Model Training & Evaluation
    print("3. Testing Model Dynamic Training & Evaluation...")
    model_cfg = OmegaConf.create({
        "class_name": "sklearn.linear_model.LogisticRegression",
        "params": {"C": 1.0, "random_state": 42}
    })
    y = [1, 0]
    model = train_model(X, y, model_cfg)
    acc, report_dict, report_str = evaluate_model(model, X, y)
    print(f"   Accuracy: {acc}")
    assert acc == 1.0, f"Expected accuracy 1.0, got {acc}"

    # 4. Test BM25 Search
    print("4. Testing BM25 Search Retriever with sentiment filtering...")
    corpus = [
        "This product is amazing and wonderful",
        "Absolutely terrible and a complete waste",
        "Great experience, highly recommend",
        "Worst purchase ever, total disappointment",
        "Love it, works perfectly",
    ]
    labels = [1, 0, 1, 0, 1]
    searcher = BM25Searcher(corpus=corpus, labels=labels)

    # Test unfiltered search
    all_results = searcher.search("product experience", n=5)
    assert len(all_results) > 0, "Expected results from unfiltered search"

    # Test positive-only filter
    pos_results = searcher.search("product experience", n=5, sentiment_filter=1)
    assert all(r["label"] == 1 for r in pos_results), "All results should be positive"

    # Test negative-only filter
    neg_results = searcher.search("terrible waste", n=5, sentiment_filter=0)
    assert all(r["label"] == 0 for r in neg_results), "All results should be negative"

    print(f"   Top positive: {pos_results[0]['document']}")
    print(f"   Top negative: {neg_results[0]['document']}")

    print("\n[SUCCESS] All component tests passed!")


if __name__ == "__main__":
    test_components()
