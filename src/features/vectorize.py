import bm25s
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from omegaconf import DictConfig


class BM25Vectorizer:
    def __init__(self, max_features=5000, k1=1.5, b=0.75):
        self.tfidf = TfidfVectorizer(max_features=max_features, norm=None, smooth_idf=False)
        self.k1 = k1
        self.b = b
        self.avg_doc_len_ = None

    def fit_transform(self, raw_documents, y=None):
        X_tfidf = self.tfidf.fit_transform(raw_documents)
        doc_lens = X_tfidf.sum(axis=1).A1
        self.avg_doc_len_ = doc_lens.mean() if len(doc_lens) > 0 else 1.0
        
        X_bm25 = csr_matrix(X_tfidf.copy())
        for i in range(X_bm25.shape[0]):
            start, end = X_bm25.indptr[i], X_bm25.indptr[i+1]
            tf = X_bm25.data[start:end]
            len_norm = 1.0 - self.b + self.b * (doc_lens[i] / self.avg_doc_len_)
            X_bm25.data[start:end] = (tf * (self.k1 + 1)) / (tf + self.k1 * len_norm)
        return X_bm25

    def transform(self, raw_documents):
        X_tfidf = self.tfidf.transform(raw_documents)
        doc_lens = X_tfidf.sum(axis=1).A1
        
        X_bm25 = csr_matrix(X_tfidf.copy())
        for i in range(X_bm25.shape[0]):
            start, end = X_bm25.indptr[i], X_bm25.indptr[i+1]
            tf = X_bm25.data[start:end]
            len_norm = 1.0 - self.b + self.b * (doc_lens[i] / self.avg_doc_len_)
            X_bm25.data[start:end] = (tf * (self.k1 + 1)) / (tf + self.k1 * len_norm)
        return X_bm25

    def get_feature_names_out(self):
        return self.tfidf.get_feature_names_out()


def vectorize_data(cleaned_corpus: list, vectorizer_cfg: DictConfig, trained_vectorizer=None):
    """
    Standardized factory function for text representations.
    """
    method = vectorizer_cfg.method
    max_features = vectorizer_cfg.get("max_features", 5000)

    if trained_vectorizer is not None:
        # Transform mode using trained vectorizer
        return trained_vectorizer.transform(cleaned_corpus)

    # Fit-transform mode
    print(f"Vectorizing corpus using method: {method}...")

    if method == "bow":
        vectorizer = CountVectorizer(max_features=max_features)
        X_matrix = vectorizer.fit_transform(cleaned_corpus)
        return X_matrix, vectorizer

    elif method == "tfidf":
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_matrix = vectorizer.fit_transform(cleaned_corpus)
        return X_matrix, vectorizer

    elif method == "bm25_features":
        k1 = vectorizer_cfg.get("k1", 1.5)
        b = vectorizer_cfg.get("b", 0.75)
        vectorizer = BM25Vectorizer(max_features=max_features, k1=k1, b=b)
        X_matrix = vectorizer.fit_transform(cleaned_corpus)
        return X_matrix, vectorizer

    else:
        raise ValueError(f"Unknown vectorization method: {method}")
