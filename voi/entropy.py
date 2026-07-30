"""
Матчасть: базовые статистики корпуса + условная энтропия глифов H_{2|1}.

H(C2 | C1) = H(C1,C2) - H(C1)   [по парам соседних глифов в потоке символов]

В статье: H₂|₁ = 2.07 бита у Войнича (живые языки 3.0+).

Замечание о потоке глифов: считаем пары соседних символов ВНУТРИ слов
(границы слов не разрывают биграмму глифов) - это даёт нижнюю оценку
предсказуемости. Альтернатива - включать пробел как отдельный символ;
привожу оба варианта для прозрачности.
"""
from __future__ import annotations
import collections
import math


def char_ngram_counts(tokens: list[str], n: int, sep=" ") -> collections.Counter:
    """Подсчёт n-грамм глифов по слитому потоку (с разделителем sep)."""
    s = sep.join(tokens)
    return collections.Counter(s[i:i + n] for i in range(len(s) - n + 1))


def _entropy_from_counts(counter: collections.Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    h = 0.0
    for cnt in counter.values():
        p = cnt / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def conditional_entropy_chars(tokens: list[str], sep=" ") -> float:
    """H(C2|C1) в битах по парам соседних глифов."""
    uni = char_ngram_counts(tokens, 1, sep)
    bi = char_ngram_counts(tokens, 2, sep)
    H1 = _entropy_from_counts(uni)
    H2 = _entropy_from_counts(bi)
    return H2 - H1


def corpus_stats(lines_df):
    """Базовые статистики: токены, уникальные, глифы, распределение по секциям."""
    toks = [t for toks in lines_df["tokens"] for t in toks]
    chars = collections.Counter()
    for t in toks:
        chars.update(t)
    by_section = {}
    for sec, grp in lines_df.groupby("section"):
        st = [t for toks in grp["tokens"] for t in toks]
        by_section[sec] = {"lines": len(grp), "tokens": len(st), "unique": len(set(st))}
    return {
        "n_lines": len(lines_df),
        "n_tokens": len(toks),
        "n_unique": len(set(toks)),
        "n_glyphs": len(chars),
        "glyph_counts": chars,
        "by_section": by_section,
    }


def run(lines_df, verbose=True) -> dict:
    toks = [t for toks in lines_df["tokens"] for t in toks]
    h_no_sep = conditional_entropy_chars(toks, sep="")  # только внутри слов
    h_with_sep = conditional_entropy_chars(toks, sep=" ")  # пробел = граница слова
    st = corpus_stats(lines_df)
    if verbose:
        print("=== МАТЧАСТЬ ===")
        print(f"строк (текстовых): {st['n_lines']}")
        print(f"токенов: {st['n_tokens']}  (статья ~37700)")
        print(f"уникальных токенов: {st['n_unique']}  (статья ~8000)")
        print(f"глифов: {st['n_glyphs']}  (статья 29)")
        print(f"H(C2|C1) без разделителя: {h_no_sep:.3f} бит")
        print(f"H(C2|C1) с пробелом:      {h_with_sep:.3f} бит")
        print(f"  статья H₂|₁ = 2.07 бит; живые языки 3.0+")
        print()
        print("по секциям:")
        for sec, d in sorted(st["by_section"].items()):
            print(f"  {sec:10s}: lines={d['lines']:5d} tokens={d['tokens']:6d} unique={d['unique']:5d}")
    return {
        "n_lines": st["n_lines"], "n_tokens": st["n_tokens"], "n_unique": st["n_unique"],
        "n_glyphs": st["n_glyphs"],
        "H_2_1_no_sep": h_no_sep, "H_2_1_with_sep": h_with_sep,
        "by_section": st["by_section"],
    }


if __name__ == "__main__":
    from voi.parse_ivtff import parse_ivtff, add_n_token_columns
    ld, _ = parse_ivtff(str(common.data_path("voynich")))
    ld = add_n_token_columns(ld)
    txt = ld[ld["n_tokens"] > 0]
    run(txt)
