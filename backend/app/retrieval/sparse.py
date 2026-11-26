"""BM25 sparse encoder for Qdrant (E1).

Document side: BM25 term-frequency saturation, computed here.
Corpus side: IDF, computed by Qdrant itself (sparse vector with Modifier.IDF),
so adding documents never requires re-encoding the others.
Query side: weight 1 per unique term.

Arabic normalization matters: without it "المؤسسة" / "الموسسة" / "مؤسسة" never match.
"""
import hashlib
import re
import unicodedata
from collections import Counter

from qdrant_client import models

ARABIC_DIACRITICS = re.compile(r"[ً-ْٰـ]")  # harakat, superscript alef, tatweel
TOKEN = re.compile(r"[\w؀-ۿ]+", re.UNICODE)

STOPWORDS = {
    # fr
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "l", "et", "ou", "en", "au", "aux",
    "a", "à", "est", "sont", "pour", "par", "sur", "dans", "que", "qui", "quel", "quelle",
    "quels", "quelles", "ce", "cette", "ces", "se", "sa", "son", "ses", "il", "elle", "ne", "pas",
    "plus", "avec", "être", "doit", "peut",
    # en
    "the", "of", "and", "or", "to", "in", "is", "are", "for", "on", "with", "what", "which",
    "an", "by", "be", "it", "this", "that",
    # ar (after normalization)
    "في", "من", "علي", "الي", "عن", "ان", "او", "هو", "هي", "ما", "التي", "الذي", "هذا", "هذه",
    "مع", "كل", "ذلك", "تلك",
    # darija
    "باش", "متاع", "فما", "شنوة", "كيفاش", "قداش",
}

ARABIC_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")


def normalize_arabic(text: str) -> str:
    text = ARABIC_DIACRITICS.sub("", text)
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي").replace("ة", "ه")
    return text


def strip_latin_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def light_stem(token: str) -> str:
    """Very light stemming: Arabic article prefixes, French/English plural 's'."""
    if re.match(r"[؀-ۿ]", token):
        for p in ARABIC_PREFIXES:
            if token.startswith(p) and len(token) - len(p) >= 2:
                return token[len(p):]
        return token
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    text = normalize_arabic(strip_latin_accents(text.lower()))
    tokens = []
    for tok in TOKEN.findall(text):
        tok = tok.strip("_")
        if not tok or tok in STOPWORDS or (len(tok) < 2 and not tok.isdigit()):
            continue
        tok = light_stem(tok)
        if tok not in STOPWORDS:
            tokens.append(tok)
    return tokens


def token_id(token: str) -> int:
    """Stable across runs and machines (Python's hash() is salted per process)."""
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "little")


class BM25Encoder:
    def __init__(self, k1: float = 1.2, b: float = 0.75, avg_len: float = 256.0):
        self.k1, self.b, self.avg_len = k1, b, avg_len

    def encode_document(self, text: str) -> models.SparseVector:
        tokens = tokenize(text)
        tf = Counter(tokens)
        dl = len(tokens) or 1
        norm = self.k1 * (1 - self.b + self.b * dl / self.avg_len)
        weights = {}
        for tok, f in tf.items():
            idx = token_id(tok)
            weights[idx] = weights.get(idx, 0.0) + f * (self.k1 + 1) / (f + norm)
        return models.SparseVector(indices=list(weights), values=list(weights.values()))

    def encode_query(self, text: str) -> models.SparseVector:
        ids = sorted({token_id(t) for t in tokenize(text)})
        return models.SparseVector(indices=ids, values=[1.0] * len(ids))
