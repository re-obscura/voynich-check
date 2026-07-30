"""
Общие утилиты: токенизация, n-граммы, JSD, TF-IDF, helpers.
Контракт на данные: lines_df из parse_ivtff со столбцом 'tokens' (list[str]).
"""
from __future__ import annotations
import math
import collections
from pathlib import Path
import numpy as np
from typing import Iterable, Sequence


# ---------- пути к данным ----------

#: корень проекта (каталог, содержащий README.md и каталог data/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

#: файлы корпуса и контрольных текстов (телефонный справочник имён)
DATA_FILES = {
    "voynich": DATA_DIR / "IT2a-n.txt",
    "kjv":     DATA_DIR / "kjv.txt",
    "moby":    DATA_DIR / "moby.txt",
    "paradise": DATA_DIR / "paradise.txt",
    "caesar":  DATA_DIR / "caesar.txt",
}


def data_path(name: str) -> Path:
    """Возвращает путь к файлу данных по ключу из DATA_FILES."""
    return DATA_FILES[name]


# ---------- токены и глифы ----------

def all_tokens(lines_tokens: Iterable[Sequence[str]]) -> list[str]:
    out = []
    for toks in lines_tokens:
        out.extend(toks)
    return out


def char_stream(tokens: Sequence[str]) -> str:
    """Слитый поток глифов: слова разделены пробелом (граница слова)."""
    return " ".join(tokens)


# ---------- распределения ----------

def freq_dist(tokens: Sequence[str]) -> dict[str, float]:
    """Относительное частотное распределение слов."""
    if not tokens:
        return {}
    c = collections.Counter(tokens)
    n = sum(c.values())
    return {w: v / n for w, v in c.items()}


def char_freq_dist(tokens: Sequence[str], sep=" ") -> dict[str, float]:
    """Частотное распределение глифов (с опц. символом границы слова)."""
    s = sep.join(tokens)
    c = collections.Counter(s)
    n = sum(c.values())
    return {ch: v / n for ch, v in c.items()}


# ---------- дивергенция Йенсена-Шеннона ----------

def _kl(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    s = 0.0
    for k in keys:
        pk = p.get(k, 0.0)
        qk = max(q.get(k, 1e-12), 1e-12)
        if pk > 0:
            s += pk * math.log2(pk / qk)
    return s


def js_divergence(p: dict, q: dict) -> float:
    """
    JSD(p,q) в битах. Использует m=(p+q)/2 и половинные веса.
    Возвращает значение в [0,1].
    """
    keys = set(p) | set(q)
    m = {}
    for k in keys:
        m[k] = 0.5 * (p.get(k, 0.0) + q.get(k, 0.0))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def js_distance(p: dict, q: dict) -> float:
    """JS distance = sqrt(JSD). В [0,1]."""
    return math.sqrt(js_divergence(p, q))


# ---------- n-граммы ----------

def ngrams(tokens: Sequence[str], n: int) -> list[tuple]:
    if n == 1:
        return [(t,) for t in tokens]
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def ngram_set(tokens: Sequence[str], n: int) -> set:
    return set(ngrams(tokens, n))


# ---------- TF-IDF / частотные векторы ----------

def build_vocab(lists_of_tokens: list[list[str]], min_count: int = 1) -> list[str]:
    c = collections.Counter()
    for toks in lists_of_tokens:
        c.update(toks)
    vocab = [w for w, cnt in c.items() if cnt >= min_count]
    vocab.sort()
    return vocab


def count_matrix(lists_of_tokens: list[list[str]], vocab: list[str]) -> np.ndarray:
    """Матрица счётов (rows = документы, cols = vocab)."""
    idx = {w: j for j, w in enumerate(vocab)}
    M = np.zeros((len(lists_of_tokens), len(vocab)), dtype=np.float64)
    for i, toks in enumerate(lists_of_tokens):
        for t in toks:
            j = idx.get(t)
            if j is not None:
                M[i, j] += 1.0
    return M


def tfidf_matrix(counts: np.ndarray) -> np.ndarray:
    """Стандартный TF-IDF с L2-нормализацией строк."""
    df = np.sum(counts > 0, axis=0)
    n_docs = counts.shape[0]
    idf = np.log((1 + n_docs) / (1 + df)) + 1.0
    tf = np.log1p(counts)
    X = tf * idf[None, :]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


# ---------- разные хелперы ----------

def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def ttr(tokens: Sequence[str]) -> float:
    """Type-Token Ratio (лексическое разнообразие)."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def jaccard(a: Iterable, b: Iterable) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
