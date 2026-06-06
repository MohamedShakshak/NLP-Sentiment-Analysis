import re
import html
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from omegaconf import DictConfig


class TextNormalizer:
    def __init__(self, preprocessing_config: DictConfig = None):
        """
        Initializes the normalizer using a configuration dictionary/DictConfig.
        Falls back to standard defaults if keys are missing.
        """
        # If no config is passed, default to an empty dictionary
        cfg = preprocessing_config if preprocessing_config is not None else {}

        # Read parameters from config with safe fallbacks
        self.lowercase = cfg.get("lowercase", True)
        self.remove_stopwords = cfg.get("remove_stopwords", True)
        self.method = cfg.get("method", "lemmatization")
        self.handle_social_tokens = cfg.get("handle_social_tokens", False)
        self.expand_contractions = cfg.get("expand_contractions", False)
        self.strip_html = cfg.get("strip_html", False)
        self.collapse_elongated = cfg.get("collapse_elongated", False)

        # Proactively check and download necessary NLTK resources
        for resource in ["stopwords", "wordnet", "omw-1.4"]:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        self.contraction_map = {
            "can't": "cannot", "won't": "will not", "idk": "i do not know", 
            "dont": "do not", "cant": "cannot", "im": "i am"
        }

    def _clean_single_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
            
        # 1. HTML Stripping (Prioritize before regexes)
        if self.strip_html:
            text = html.unescape(text)
            text = re.sub(r'<.*?>', ' ', text)
            
        # 2. Handle Social Tokens (Twitter specific)
        if self.handle_social_tokens:
            text = re.sub(r'@\S+', '[USER]', text)
            text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
            
        # 3. Collapse elongated words (e.g., coool -> cool)
        if self.collapse_elongated:
            text = re.sub(r'(.)\1+', r'\1\1', text)
            
        if self.lowercase:
            text = text.lower()
            
        # Strip general punctuation here if necessary, but keep structure for social tokens
        if not self.handle_social_tokens:
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
        # 4. Tokenize and execute standard mappings
        tokens = text.split()
        
        if self.expand_contractions:
            tokens = [self.contraction_map.get(word, word) for word in tokens]
            
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]
            
        # 5. Normalization Method (Stemming vs Lemmatization vs None)
        if self.method == "lemmatization":
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        elif self.method == "stemming":
            tokens = [self.stemmer.stem(word) for word in tokens]
            
        return " ".join(tokens)

    def transform(self, texts: pd.Series) -> list:
        print(f"Running text normalization pipeline...")
        return [self._clean_single_text(text) for text in texts]
