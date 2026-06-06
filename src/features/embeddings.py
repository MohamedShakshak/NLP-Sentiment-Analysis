import pickle
from pathlib import Path

from gensim.models import Word2Vec


def train_embeddings(cleaned_corpus: list, vector_size: int = 100,
                     window: int = 5, min_count: int = 2,
                     workers: int = 4, epochs: int = 5) -> Word2Vec:
    """
    Train a Word2Vec model on a pre-cleaned corpus (list of strings).
    """
    tokenized = [doc.split() for doc in cleaned_corpus if doc.strip()]
    print(f"Training Word2Vec on {len(tokenized):,} documents "
          f"(vector_size={vector_size}, window={window}, epochs={epochs})...")
    model = Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        seed=42,
    )
    return model


def save_embeddings(model: Word2Vec, path: str) -> None:
    """Save the Word2Vec model to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    print(f"Word2Vec model saved to: {path}")


def load_embeddings(path: str) -> Word2Vec:
    """Load a saved Word2Vec model from disk."""
    return Word2Vec.load(path)


def get_vector(model: Word2Vec, word: str):
    """Return the embedding vector for a single word (or None if OOV)."""
    if word in model.wv:
        return model.wv[word]
    return None


def most_similar(model: Word2Vec, word: str, n: int = 10):
    """Return n most similar words to the given word."""
    if word not in model.wv:
        return []
    return model.wv.most_similar(word, topn=n)
