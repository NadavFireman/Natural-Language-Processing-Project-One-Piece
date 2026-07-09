"""
text_features.py
One Piece NLP Project — text (NLP) feature layer.

The core NLP stage. Turns each character's clean_text description into numeric
features through three complementary representations, plus a sentiment signal
and a Zipf's-law corpus analysis:

  1. preprocess_text  - lowercase, tokenize, drop stopwords, lemmatize (NLTK)
  2. TF-IDF           - sparse vectors reduced with TruncatedSVD
  3. MiniLM embeddings- optional extension: implemented with the same
                        train-only leakage control, NOT activated in the
                        final runs (use_embeddings=False)
  4. add_sentiment    - a "threat" lexicon score from the text
  5. zipf_analysis    - corpus word-frequency / Zipf check (for the report)

Leakage control: TF-IDF and SVD are FIT ON TRAIN ONLY and used to transform
val/test. The notebook passes the train texts. The MiniLM pipeline
(embed -> SVD fit on train -> transform) is fully implemented but was NOT
activated in the final runs; no reported number uses it. Measuring its
contribution is future work.
"""

import re
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def _ensure_nltk():
    """Download the NLTK resources this module needs, once, if missing."""
    for res, path in [("punkt", "tokenizers/punkt"),
                      ("punkt_tab", "tokenizers/punkt_tab"),
                      ("stopwords", "corpora/stopwords"),
                      ("wordnet", "corpora/wordnet"),
                      ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, quiet=True)


_ensure_nltk()

# ----------------------------------------------------------------------
# 1. Preprocessing
# ----------------------------------------------------------------------
_LEMMATIZER = WordNetLemmatizer()
_STOP = set(stopwords.words("english"))
# domain stop-words: appear in nearly every article, carry no signal
_DOMAIN_STOP = {"one", "piece", "chapter", "episode", "manga", "anime",
                "viz", "page", "vol", "volume", "also", "would", "could",
                "later", "however", "though", "first", "time", "see", "name"}
_STOP |= _DOMAIN_STOP


def preprocess_text(text):
    """Lowercase, tokenize, drop stop-words and short tokens, lemmatize.
    Returns a single cleaned string (space-joined lemmas)."""
    if not isinstance(text, str) or not text:
        return ""
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    out = []
    for tok in word_tokenize(text):
        if len(tok) < 3 or tok in _STOP:
            continue
        out.append(_LEMMATIZER.lemmatize(tok))
    return " ".join(out)


def add_clean_tokens(df, text_col="clean_text", out_col="text_clean"):
    """Add a preprocessed-text column used by TF-IDF and Zipf analysis."""
    df = df.copy()
    df[out_col] = df[text_col].apply(preprocess_text)
    return df


# ----------------------------------------------------------------------
# 2. TF-IDF  (fit on train only, then reduce with SVD)
# ----------------------------------------------------------------------
def fit_tfidf(train_texts, max_features=3000, ngram_range=(1, 2)):
    """Fit a TF-IDF vectorizer on the training texts only."""
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                          min_df=3, max_df=0.85, sublinear_tf=True)
    vec.fit(train_texts)
    return vec


def fit_tfidf_svd(vec, train_texts, n_components=50, seed=42):
    """Fit a TruncatedSVD to compress sparse TF-IDF into dense columns."""
    X = vec.transform(train_texts)
    n_components = min(n_components, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    svd.fit(X)
    return svd


def transform_tfidf(vec, svd, texts, prefix="tfidf_"):
    """Transform texts -> reduced dense TF-IDF DataFrame."""
    X = svd.transform(vec.transform(texts))
    cols = [f"{prefix}{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols, index=getattr(texts, "index", None))


def top_tfidf_terms(vec, n=25):
    """Highest-IDF-weighted vocabulary terms, for inspection in the report."""
    idf = vec.idf_
    vocab = np.array(vec.get_feature_names_out())
    order = np.argsort(idf)
    return list(vocab[order][:n]), list(vocab[order][-n:])


# ----------------------------------------------------------------------
# 3. MiniLM embeddings  (optional extension - not activated in the final runs)
# ----------------------------------------------------------------------
def load_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Load the MiniLM sentence-transformer model (call once)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def embed_texts(model, texts, max_chars=2000, batch_size=32):
    """Encode texts with MiniLM. We pass the opening `max_chars` of each
    description (the lead summarizes the character) to stay within the model's
    context and keep it fast. Returns an (n, 384) float array."""
    clipped = [str(t)[:max_chars] for t in texts]
    emb = model.encode(clipped, batch_size=batch_size,
                       show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(emb)


def fit_embedding_svd(train_emb, n_components=30, seed=42):
    """Reduce 384-dim embeddings to a handful of dense columns (fit on train
    only, like the TF-IDF SVD)."""
    n_components = min(n_components, train_emb.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    svd.fit(train_emb)
    return svd


def transform_embeddings(svd, emb, prefix="emb_", index=None):
    """Reduce a full embedding matrix to a DataFrame of dense columns."""
    X = svd.transform(emb)
    cols = [f"{prefix}{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols, index=index)


# ----------------------------------------------------------------------
# 4. Sentiment / "threat" lexicon score
# ----------------------------------------------------------------------
# A small, transparent lexicon. Bounty reflects how dangerous the World
# Government perceives a character to be, so we count threat/menace language.
_THREAT_WORDS = {
    "dangerous", "danger", "feared", "fear", "threat", "threatening", "deadly",
    "powerful", "power", "strongest", "strong", "fearsome", "terrifying",
    "menace", "menacing", "brutal", "ruthless", "violent", "destroy",
    "destruction", "kill", "killed", "death", "war", "monster", "demon",
    "infamous", "notorious", "feared", "rampage", "massacre", "tyrant",
    "overwhelming", "devastating", "merciless", "savage", "conqueror",
}
_POSITIVE_WORDS = {
    "kind", "gentle", "friendly", "loyal", "brave", "hero", "heroic", "good",
    "honest", "noble", "peaceful", "calm", "cheerful", "protect", "save",
}


def threat_score(text):
    """Net 'threat' density: (threat - positive) word hits per 1000 tokens.
    Higher = the article frames the character as more dangerous."""
    if not isinstance(text, str) or not text:
        return 0.0
    toks = re.findall(r"[a-z]+", text.lower())
    if not toks:
        return 0.0
    t = sum(1 for w in toks if w in _THREAT_WORDS)
    p = sum(1 for w in toks if w in _POSITIVE_WORDS)
    return 1000.0 * (t - p) / len(toks)


def add_sentiment(df, text_col="clean_text", out_col="threat_score"):
    """Add the threat-lexicon score column."""
    df = df.copy()
    df[out_col] = df[text_col].apply(threat_score)
    return df


# ----------------------------------------------------------------------
# 5. Zipf's-law corpus analysis (for the report / a visualization)
# ----------------------------------------------------------------------
def zipf_analysis(texts, top_n=30):
    """Return (rank, freq, word) arrays over the corpus for a Zipf plot,
    plus the most common words. Uses the preprocessed text."""
    from collections import Counter
    counter = Counter()
    for t in texts:
        counter.update(str(t).split())
    most = counter.most_common()
    freqs = np.array([c for _, c in most], dtype=float)
    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    top = most[:top_n]
    return ranks, freqs, top