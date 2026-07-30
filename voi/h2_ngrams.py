"""
Гипотеза 2: в тексте есть готовые шаблонные блоки.

Версия: писец собирал текст из повторяющихся кусков (таблица слогов / заготовки),
особенно в bio и recipes.

Проверки (заявленные):
  1. Должны найтись длинные повторяющиеся n-граммы (3-4 слова) на РАЗНЫХ тетрадях.
     Статья: ни один 4-граммный блок не повторяется хотя бы на 3 разных тетрадях,
     ни в bio, ни в recipes, ни в herbal.
  2. Доля соседних дублей биграмм (X X) одинакова по всем разделам (0.7-1.1%).
     Это 'aiin aiin' - просто частые слова рядом.
  3. Низкое лексическое разнообразие bio (TTR=0.12) vs herbal (0.35) объясняется
     маленьким словарём раздела, а не шаблонным повторением.

Вывод статьи: гипотеза не подтвердилась. TTR-разница = размер словаря, не шаблон.
"""
from __future__ import annotations
import collections
from voi import common


def ngram_repeats_across_quires(lines_df, n=4, min_quires=3):
    """
    Сколько n-грамм повторяются на >= min_quires разных тетрадях.
    Возвращаем: для каждой секции - (кол-во n-грамм на >=min_quires тетрадях,
    общее кол-во уникальных n-грамм, примеры).
    """
    results = {}
    for section in ["bio", "recipes", "herbal"]:
        sec = lines_df[lines_df["section"] == section]
        # n-грамма -> множество тетрадей, где встретилась
        ng_quires = collections.defaultdict(set)
        for toks, quire in zip(sec["tokens"], sec["quire"]):
            for ng in common.ngrams(list(toks), n):
                ng_quires[ng].add(quire)
        total_unique = len(ng_quires)
        repeated = [(ng, len(qs)) for ng, qs in ng_quires.items() if len(qs) >= min_quires]
        repeated.sort(key=lambda x: -x[1])
        results[section] = {
            "n_unique_ngrams": total_unique,
            "n_repeated_min_quires": len(repeated),
            "max_quires_for_one_ngram": max((len(qs) for qs in ng_quires.values()), default=0),
            "top": repeated[:5],
        }
    return results


def adjacent_duplicate_bigram_fraction(lines_df):
    """
    Доля биграмм вида (X, X) - соседний дубликат - среди всех биграмм, по секциям.
    Статья: 0.7-1.1% во всех разделах.
    """
    out = {}
    for section, grp in lines_df.groupby("section"):
        n_total = 0
        n_dup = 0
        for toks in grp["tokens"]:
            for a, b in common.ngrams(list(toks), 2):
                n_total += 1
                if a == b:
                    n_dup += 1
        out[section] = (n_dup / n_total * 100) if n_total else 0.0
    return out


def section_ttr(lines_df):
    """
    TTR по секциям. Внимание: TTR зависит от длины. Статья сравнивает
    bio (0.12) и herbal (0.35). Для честности приведём также TTR на
    унифицированной длине (MATTR-like: среднее TTR по окнам).
    """
    out = {}
    for section, grp in lines_df.groupby("section"):
        toks = [t for ts in grp["tokens"] for t in ts]
        out[section] = {
            "raw_ttr": common.ttr(toks),
            "n_tokens": len(toks),
            "n_unique": len(set(toks)),
        }
    return out


def mattr(tokens: list[str], window: int = 100) -> float:
    """Moving-average TTR: устойчиво к длине текста."""
    if len(tokens) <= window:
        return common.ttr(tokens)
    vals = []
    for i in range(0, len(tokens) - window + 1):
        vals.append(len(set(tokens[i:i + window])) / window)
    return sum(vals) / len(vals)


def run(lines_df, verbose=True) -> dict:
    out = {}
    # только параграфный текст для устойчивости
    P = lines_df[lines_df["tokens"].apply(len) > 0].copy()

    if verbose:
        print("=== ГИПОТЕЗА 2: шаблонные блоки ===")

    # 1. повторяющиеся 4-граммы по тетрадям
    rep = ngram_repeats_across_quires(P, n=4, min_quires=3)
    out["ngram4_across_quires"] = rep
    if verbose:
        for sec in ["bio", "recipes", "herbal"]:
            r = rep[sec]
            print(f"4-граммы [{sec}]: уникальных {r['n_unique_ngrams']}, "
                  f"на >=3 тетрадях: {r['n_repeated_min_quires']}, "
                  f"макс. тетрадей для одной n-граммы: {r['max_quires_for_one_ngram']}")
        print(f"  (статья: ни одной 4-граммы на >=3 тетрадях)")
        print()

    # также 3-граммы для контекста
    rep3 = ngram_repeats_across_quires(P, n=3, min_quires=3)
    out["ngram3_across_quires"] = rep3
    if verbose:
        for sec in ["bio", "recipes", "herbal"]:
            r = rep3[sec]
            print(f"3-граммы [{sec}]: на >=3 тетрадях: {r['n_repeated_min_quires']}, "
                  f"макс тетрадей: {r['max_quires_for_one_ngram']}")

    # 2. соседние дубли биграмм
    dup = adjacent_duplicate_bigram_fraction(P)
    out["adj_dup_bigram_pct"] = dup
    if verbose:
        print()
        print("Соседние дубли биграмм (X X), % от всех биграмм:")
        for sec in ["bio", "recipes", "herbal", "pharma", "astro"]:
            if sec in dup:
                print(f"  {sec:9s}: {dup[sec]:.2f}%")
        print(f"  (статья: 0.7-1.1% во всех разделах)")
        print()

    # 3. TTR
    ttr = section_ttr(P)
    out["ttr_raw"] = ttr
    # MATTR для сравнения при унифицированной длине
    mattr_by_sec = {}
    for section, grp in P.groupby("section"):
        toks = [t for ts in grp["tokens"] for t in ts]
        if len(toks) >= 100:
            mattr_by_sec[section] = mattr(toks, window=100)
    out["mattr_window100"] = mattr_by_sec
    if verbose:
        print("TTR по секциям:")
        for sec in ["bio", "recipes", "herbal", "pharma", "astro"]:
            if sec in ttr:
                m = mattr_by_sec.get(sec, float('nan'))
                print(f"  {sec:9s}: raw_ttr={ttr[sec]['raw_ttr']:.3f}  "
                      f"(tokens={ttr[sec]['n_tokens']:6d}, unique={ttr[sec]['n_unique']:5d})  "
                      f"MATTR(w=100)={m:.3f}")
        print(f"  (статья: bio raw 0.12 vs herbal 0.35)")
        print()
    return out


if __name__ == "__main__":
    from voi.parse_ivtff import parse_ivtff, add_n_token_columns
    ld, _ = parse_ivtff(str(common.data_path("voynich")))
    ld = add_n_token_columns(ld)
    run(ld)
