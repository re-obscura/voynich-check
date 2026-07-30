"""
Гипотеза 4: если это язык, должен быть синтаксис.

Идея: если в тексте есть настоящая грамматика, биграммы должны нести больше
информации о секции, чем список частых слов без учёта порядка.

Leave-one-folio-out классификация секции по частотному профилю:
  статья: униграммы 74.7%, биграммы 66.7%, триграммы 61.3%.
  Чем длиннее контекст, тем хуже предсказание => синтаксиса нет.

Внутри одного диалекта (Currier B): униграммы 81.4% при baseline 34.9%.

Топ-20 биграмм для разных секций почти не пересекаются (Жаккар 0.00-0.11),
но это следствие разного набора частых слов, а не устойчивых словосочетаний.

Вывод статьи: секция определяется ЧАСТОТНОСТЬЮ слов, а не их ПОРЯДКОМ.
"""
from __future__ import annotations
import collections
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from voi import common


def folio_ngram_lists(lines_df, n: int, min_df: int = 2):
    """
    Для каждого фолио: документ = список n-грамм (последовательность токенов строки).
    n=1 - униграммы (слова).
    Возвращает (folios, doc_token_lists, section_labels, currier_labels).
    """
    by_folio = collections.OrderedDict()
    meta = {}
    for folio, toks, sec, cur in zip(lines_df["folio"], lines_df["tokens"],
                                     lines_df["section"], lines_df["currier"]):
        by_folio.setdefault(folio, []).extend(toks)
        meta.setdefault(folio, (sec, cur))
    folios = list(by_folio.keys())
    docs = []
    for f in folios:
        toks = by_folio[f]
        docs.append(list(common.ngrams(list(toks), n)))
    sec_labels = [meta[f][0] for f in folios]
    cur_labels = [meta[f][1] for f in folios]
    return folios, docs, sec_labels, cur_labels


def loo_classify(docs, labels, min_df: int = 2):
    """
    Leave-one-out классификация (на уровне документа/фолио) через MultinomialNB
    на счётных n-грамм-векторах. Возвращает точность и базовую линию (most-frequent).
    """
    # vocab
    vocab = common.build_vocab(docs, min_count=min_df)
    idx = {w: i for i, w in enumerate(vocab)}
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    Xall = common.count_matrix(docs, vocab)  # (n_docs, n_vocab)
    n = Xall.shape[0]
    correct = 0
    for i in range(n):
        train = np.delete(np.arange(n), i)
        clf = MultinomialNB(alpha=0.1)
        clf.fit(Xall[train], y[train])
        pred = clf.predict(Xall[i:i + 1])[0]
        if pred == y[i]:
            correct += 1
    acc = correct / n
    # baseline = доля самого частого класса
    baseline = max(np.bincount(y)) / n
    return acc, baseline, n_classes, n


def top_ngram_jaccard(lines_df, n=2, top=20, sections=("herbal", "bio", "recipes", "pharma")):
    """
    Для каждой пары секций - Жаккар топ-N биграмм. Статья: 0.00-0.11.
    """
    sec_top = {}
    for sec in sections:
        grp = lines_df[lines_df["section"] == sec]
        ngc = collections.Counter()
        for toks in grp["tokens"]:
            for ng in common.ngrams(list(toks), n):
                ngc[ng] += 1
        topset = set(w for w, _ in ngc.most_common(top))
        sec_top[sec] = topset
    pairs = []
    secs = list(sec_top.keys())
    for i in range(len(secs)):
        for j in range(i + 1, len(secs)):
            j_val = common.jaccard(sec_top[secs[i]], sec_top[secs[j]])
            pairs.append((secs[i], secs[j], j_val))
    return pairs


def run(lines_df, verbose=True) -> dict:
    out = {}
    # только секции с достаточным числом фолио
    valid_sections = {"herbal", "bio", "recipes", "pharma", "astro"}
    df = lines_df[lines_df["section"].isin(valid_sections)].copy()

    if verbose:
        print("=== ГИПОТЕЗА 4: есть ли синтаксис (биграммы vs униграммы) ===")

    for n in (1, 2, 3):
        folios, docs, sec_labels, cur_labels = folio_ngram_lists(df, n=n)
        acc, baseline, k, nd = loo_classify(docs, sec_labels, min_df=2)
        out[f"loo_uni{n}gram"] = {"acc": acc, "baseline": baseline, "n_classes": k, "n_docs": nd}
        if verbose:
            print(f"Leave-one-folio-out [n={n}]: acc={acc*100:.1f}%  (baseline {baseline*100:.1f}%, классов {k}, фолио {nd})")
    if verbose:
        print(f"  (статья: уни 74.7%, би 66.7%, три 61.3% - чем длиннее контекст, тем хуже)")
        print()

    # внутри Currier B
    dfB = df[df["currier"] == "B"].copy()
    # оставим секции с >=3 фолио в B
    cnt = collections.Counter(dfB["section"])
    keep_sec = {s for s, c in cnt.items() if c >= 3}
    dfB = dfB[dfB["section"].isin(keep_sec)]
    if len(dfB) >= 10 and len(keep_sec) >= 2:
        folios, docs, sec_labels, cur_labels = folio_ngram_lists(dfB, n=1)
        acc, baseline, k, nd = loo_classify(docs, sec_labels, min_df=2)
        out["within_B_unigram"] = {"acc": acc, "baseline": baseline, "n_classes": k, "n_docs": nd}
        if verbose:
            print(f"Внутри Currier B [униграммы]: acc={acc*100:.1f}%  (baseline {baseline*100:.1f}%)")
            print(f"  (статья: 81.4% при baseline 34.9%)")
    else:
        if verbose:
            print("Внутри Currier B: недостаточно секций/фолио")
    if verbose:
        print()

    # Жаккар топ-20 биграмм
    pairs = top_ngram_jaccard(df, n=2, top=20)
    out["jaccard_top20_bigram"] = pairs
    if verbose:
        print("Жаккар топ-20 биграмм между секциями:")
        for a, b, jv in pairs:
            print(f"  {a:8s} vs {b:8s}: {jv:.2f}")
        print(f"  (статья: 0.00-0.11)")
        print()
    return out


if __name__ == "__main__":
    from voi.parse_ivtff import parse_ivtff, add_n_token_columns
    ld, _ = parse_ivtff(str(common.data_path("voynich")))
    ld = add_n_token_columns(ld)
    run(ld)
