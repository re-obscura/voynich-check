"""
Подготовка данных для glyph-level LM.

Контракт: переиспользуем parse_ivtff.parse_ivtff() -> lines_df. Из строк
строим один поток глифов на фолио, сохраняя структуру (строка / абзац / страница)
через спецтокены. Hold-out делаем на уровне ФОЛИО (не позиций).

Поток Войнича на фолио:
    слово.слово.слово<sep>слово...      # <sep> (= '.') отделяет слова в строке
    строка строки в абзаце склеиваются <sep> в конце каждой строки
    конец абзаца / фолио -> <eos>

Спецтокены: <pad>=0, <bos>, <eos>, <sep>. <sep> = '.' EVA-разделитель слов.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from voi.common import DATA_DIR, data_path
from voi.parse_ivtff import parse_ivtff, add_n_token_columns


# --- спецтокены ---
PAD, BOS, EOS, SEP = "<pad>", "<bos>", "<eos>", "<sep>"
SPECIAL = [PAD, BOS, EOS, SEP]
SEP_GLYPH = "."  # EVA word separator — это отдельный токен словаря, не спецтокен


# ============================================================
#  Токенизатор glyph-level (словарь строится из данных)
# ============================================================

@dataclass
class GlyphTokenizer:
    """Сопоставляет глифы -> int. Словарь строится из обучающего потока."""
    stoi: dict = field(default_factory=dict)
    itos: list = field(default_factory=list)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, glyphs: set[str], specials: list[str] | None = None) -> "GlyphTokenizer":
        specials = specials or SPECIAL
        # стабильно сортируем для воспроизводимости
        chars = sorted(g for g in glyphs if g != "")
        vocab = list(specials) + chars
        stoi = {t: i for i, t in enumerate(vocab)}
        return cls(stoi=stoi, itos=vocab)

    def encode(self, text: str) -> list[int]:
        out = []
        for ch in text:
            i = self.stoi.get(ch)
            if i is not None:
                out.append(i)
        return out

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids if 0 <= i < len(self.itos))

    def to_json(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_json(cls, d: dict) -> "GlyphTokenizer":
        itos = d["itos"]
        stoi = {t: i for i, t in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @property
    def sep_id(self) -> int:
        return self.stoi[SEP]

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]


# ============================================================
#  Поток Войнича: фолио -> строка глифов
# ============================================================

def folio_glyph_stream(folio_lines) -> str:
    """
    folio_lines: список (tokens, para_pos) для строк одного фолио (в порядке).
    Возвращает строку глифов:
      слово.слово.<sep>слово...
      строка = слова, соединённые SEP_GLYPH, в конце строки + SEP_GLYPH
      смена абзаца (para_pos==0 кроме первой строки) -> <eos>
    Здесь мы НЕ вставляем текстовые '<sep>' токены в строку — разделителем
    слов остаётся '.' (он и есть SEP_GLYPH). Спецтокены вставим при encode.
    """
    # упрощённо: используем маркеры '#' для <sep> границы строки и '|' для <eos>
    # чтобы потом заменить на спецтокены на этапе encode потока.
    raise NotImplementedError  # используем encode_folio ниже


def folio_to_id_list(folio_lines, tok: GlyphTokenizer) -> list[int]:
    """
    folio_lines: список dict с 'tokens' (list[str]) и 'para_pos' (int).
    Возвращает список id глифов для одного фолио с внедрёнными спецтокенами:
      слово.слово.        # '.' = SEP_GLYPH между словами и в конце строки
      ... <eos> ...       # на смене абзаца
    """
    ids: list[int] = []
    sep_id = tok.stoi[SEP_GLYPH]
    eos_id = tok.eos_id
    prev_para = None
    for ln in folio_lines:
        toks = ln["tokens"]
        if not toks:
            continue
        # вставка eos при смене абзаца (кроме самого начала)
        pp = ln["para_pos"]
        if prev_para is not None and pp == 0:
            ids.append(eos_id)
        prev_para = pp
        # слова через SEP_GLYPH, в конце строки ещё один SEP_GLYPH (граница строки)
        for w in toks:
            for ch in w:
                i = tok.stoi.get(ch)
                if i is not None:
                    ids.append(i)
            ids.append(sep_id)
    return ids


# ============================================================
#  Сборка датасета Войнича
# ============================================================

@dataclass
class VoynichDataset:
    tok: GlyphTokenizer
    train_ids: list[int]
    val_ids: list[int]
    test_folios: list[str]
    train_folios: list[str]
    # поблочная разбивка по фолио для анализа позиции / cross-dialect
    folio_blocks: dict   # folio -> list[int]
    folio_meta: dict     # folio -> {section, currier, scribe, para_pos-per-line...}


def _split_folios(folios: list[str], test_frac: float = 0.15, seed: int = 7) -> tuple[list[str], list[str]]:
    """Детерминированный hold-out по фолио. Возвращает (train, test)."""
    rng = np.random.default_rng(seed)
    shuffled = sorted(folios)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_frac)))
    test = sorted(shuffled[:n_test])
    train = sorted(shuffled[n_test:])
    return train, test


def build_voynich_dataset(ctx_max_glypha: int | None = None,
                          test_frac: float = 0.15,
                          seed: int = 7,
                          save_split: Path | None = None) -> VoynichDataset:
    """
    Парсит транскрипцию, строит токенизатор (словарь из всех глифов),
    разбивает фолио на train/test, собирает потоки глифов.
    """
    ld, _ = parse_ivtff(str(data_path("voynich")))
    ld = add_n_token_columns(ld)
    txt = ld[ld["n_tokens"] > 0].copy()
    # отсортируем по фолио, потом по locus_seq
    txt = txt.sort_values(["folio_num", "locus_seq"]).reset_index(drop=True)

    # --- метаданные по фолио (для cross-dialect / probing) ---
    folio_meta: dict = {}
    for f, row in zip(txt["folio"], txt[["section", "currier", "scribe"]].itertuples(index=False)):
        folio_meta.setdefault(f, {"section": row.section,
                                  "currier": row.currier, "scribe": row.scribe})

    # --- собрать все глифы для словаря ---
    all_glyphs: set[str] = set()
    for toks in txt["tokens"]:
        for w in toks:
            all_glyphs.update(w)
    all_glyphs.add(SEP_GLYPH)
    tok = GlyphTokenizer.build(all_glyphs)

    # --- поблочная разбивка по фолио ---
    folio_blocks: dict[str, list[int]] = {}
    for f, grp in txt.groupby("folio"):
        lines = [{"tokens": t, "para_pos": p}
                 for t, p in zip(grp["tokens"], grp["para_pos"])]
        folio_blocks[str(f)] = folio_to_id_list(lines, tok)

    all_folios = sorted(folio_blocks.keys())
    train_folios, test_folios = _split_folios(all_folios, test_frac, seed)

    train_ids: list[int] = []
    eos = tok.eos_id
    for f in train_folios:
        train_ids.extend(folio_blocks[f])
        train_ids.append(eos)
    # val = test-фолио
    val_ids: list[int] = []
    for f in test_folios:
        val_ids.extend(folio_blocks[f])
        val_ids.append(eos)

    ds = VoynichDataset(
        tok=tok, train_ids=train_ids, val_ids=val_ids,
        test_folios=test_folios, train_folios=train_folios,
        folio_blocks=folio_blocks, folio_meta=folio_meta,
    )
    if save_split:
        save_split.write_text(json.dumps({
            "train_folios": train_folios, "test_folios": test_folios,
            "vocab": tok.to_json(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return ds


# ============================================================
#  Контроли: Gutenberg (обрезанные до длины Войнича) + shuffle-null
# ============================================================

def load_gutenberg_chars(path: str, max_chars: int | None = None) -> str:
    """
    Чистый загрузчик Gutenberg: вырезать хедер/футер, lowercase, оставить
    буквы и пробел. Возвращает сплошной поток символов (пробел — граница слова).
    """
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    start = txt.find("*** START OF")
    end = txt.find("*** END OF")
    if start != -1 and end != -1:
        body = txt[start:end]
        # убрать строку-маркер START и всё до конца той строки
        nl = body.find("\n")
        if nl != -1:
            body = body[nl + 1:]
    else:
        body = txt
    # оставим буквы и пробелы, схлопнем множественные пробелы
    cleaned = re.sub(r"[^a-zA-Z ]", " ", body).lower()
    cleaned = re.sub(r" +", " ", cleaned).strip()
    if max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned


def build_control_dataset(name: str, target_chars: int,
                           text_path: str | None = None,
                           seed: int = 7) -> tuple[GlyphTokenizer, list[int]]:
    """
    name: 'kjv' | 'moby' | 'paradise' | 'caesar' -> природный язык
          'shuffle' -> перемешанный поток Войнича (null)
    Возвращает (tokenizer, ids) для ОДНОЙ контрольной модели.
    """
    if name == "shuffle":
        # перемешиваем глифы потока Войнича (с тем же словарём)
        ds_v = build_voynich_dataset()
        ids = list(ds_v.train_ids)
        rng = np.random.default_rng(seed)
        rng.shuffle(ids)
        return ds_v.tok, ids

    assert text_path is not None
    chars = load_gutenberg_chars(text_path, max_chars=target_chars)
    glyphs = set(chars)
    glyphs.add(SEP_GLYPH)
    tok = GlyphTokenizer.build(glyphs)
    ids = tok.encode(chars)
    return tok, ids


# ============================================================
#  DataLoader: случайные окна длины ctx из потока id
# ============================================================

def make_batches(ids: list[int], ctx: int, batch_size: int, device,
                 n_batches: int, seed: int = 0) -> torch.Tensor:
    """Случайные сэмплы x/y из потока (стандартный nanoGPT-батч)."""
    rng = np.random.default_rng(seed)
    data = np.array(ids, dtype=np.int64)
    ix = rng.integers(0, len(data) - ctx - 1, size=n_batches * batch_size)
    xs = np.stack([data[i:i + ctx] for i in ix])
    ys = np.stack([data[i + 1:i + 1 + ctx] for i in ix])
    X = torch.tensor(xs, dtype=torch.long, device=device)
    Y = torch.tensor(ys, dtype=torch.long, device=device)
    return X, Y


def evaluate_loader(ids: list[int], ctx: int, batch_size: int, device,
                     seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Непересекающиеся окна для честной оценки perplexity на всём потоке."""
    data = np.array(ids, dtype=np.int64)
    n = (len(data) - 1) // ctx
    xs, ys = [], []
    for k in range(n):
        i = k * ctx
        xs.append(data[i:i + ctx])
        ys.append(data[i + 1:i + 1 + ctx])
    X = torch.tensor(np.stack(xs), dtype=torch.long, device=device)
    Y = torch.tensor(np.stack(ys), dtype=torch.long, device=device)
    return X, Y


if __name__ == "__main__":
    ds = build_voynich_dataset()
    print("vocab:", ds.tok.itos)
    print("vocab size:", ds.tok.vocab_size)
    print("train folios:", len(ds.train_folios), "test folios:", len(ds.test_folios))
    print("train glyphs:", len(ds.train_ids), "val glyphs:", len(ds.val_ids))
    print("sample decode:", ds.tok.decode(ds.train_ids[200:280]))
