import bm25s
import numpy as np


class BM25Searcher:
    """
    BM25 retrieval index with optional sentiment filtering.
    """
    def __init__(self, corpus=None, labels=None):
        self.retriever = None
        self.corpus = None
        self.labels = None
        if corpus is not None:
            self.fit(corpus, labels)

    def fit(self, corpus: list, labels: list = None):
        """
        Index a corpus of documents.

        Args:
            corpus: list of raw text strings.
            labels: optional list of int labels (0=negative, 1=positive),
                    same length as corpus. Used for sentiment filtering.
        """
        self.corpus = list(corpus)
        self.labels = list(labels) if labels is not None else None

        tokenized_corpus = [doc.split() for doc in self.corpus]
        self.retriever = bm25s.BM25()
        self.retriever.index(tokenized_corpus)

    def search(self, query: str, n: int = 5, sentiment_filter: int = None) -> list:
        """
        Search the index for documents matching the query.

        Args:
            query: search string.
            n: number of results to return.
            sentiment_filter: if 0 or 1, only return docs with that label.
                              Requires labels to have been provided at fit() time.

        Returns:
            List of dicts with keys: document, index, score, label (if available).
        """
        if self.retriever is None:
            raise ValueError("BM25 retriever has not been fitted.")

        tokenized_query = [query.split()]   # bm25s expects [[token, ...]]

        # Retrieve more candidates if filtering is requested
        candidate_k = min(len(self.corpus), n * 5 if sentiment_filter is not None else n)
        results, scores = self.retriever.retrieve(tokenized_query, k=candidate_k)

        output = []
        for idx, score in zip(results[0], scores[0]):
            label = self.labels[idx] if self.labels is not None else None

            if sentiment_filter is not None and label != sentiment_filter:
                continue

            entry = {
                "document": self.corpus[idx],
                "index": int(idx),
                "score": float(score),
            }
            if label is not None:
                entry["label"] = int(label)

            output.append(entry)
            if len(output) >= n:
                break

        return output
