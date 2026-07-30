"""
Гипотеза 3: различия между секциями объясняются просто разными писцами.

Проверки (заявленные):
  Кластеризация (UMAP + K-means, silhouette, ARI vs меток):
    - Currier A/B:    silhouette +0.15, ARI до 0.64 (при делении на 2 кластера)
    - метка 'секция': отрицательный silhouette (хуже случайного)
    - метка 'писец':  отрицательный silhouette
    Внутри диалекта B: bio выделяется (ARI ~ 0.25). Внутри A - ничего.
  Тест перестановок соседних тетрадей (10 000 итераций):
    herbal p = 0.80, bio p = 0.55 - соседние тетради не отличаются от случайных пар.

Вывод статьи: реально кластеризуется только Currier A/B.
"""
from __future__ import annotations
import collections
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

from voi import common


def folio_vectors(lines_df, min_df=2):
    """
    Вектор фолио = TF-IDF частотного профиля слов этой страницы.
    Возвращает (folio_list, X, meta_df) где meta = фолио-метки.
    """
    # собрать токены по фолио
    by_folio = collections.OrderedDict()
    for folio, toks in zip(lines_df["folio"], lines_df["tokens"]):
        by_folio.setdefault(folio, []).extend(toks)
    folios = list(by_folio.keys())
    lists = [by_folio[f] for f in folios]

    vocab = common.build_vocab(lists, min_count=min_df)
    counts = common.count_matrix(lists, vocab)
    X = common.tfidf_matrix(counts)
    return folios, X, by_folio


def cluster_eval(X, labels, n_clusters=2, seed=42, use_umap=True, umap_neighbors=15):
    """
    K-means на (UMAP-)TF-IDF. Возвращает silhouette (по X или UMAP) и ARI vs labels.
    """
    if use_umap:
        import umap
        # устойчивые параметры UMAP
        reducer = umap.UMAP(n_neighbors=umap_neighbors, n_components=5,
                            metric="cosine", random_state=seed)
        Xred = reducer.fit_transform(X)
    else:
        Xred = X

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    pred = km.fit_predict(Xred)
    sil = silhouette_score(Xred, pred) if len(set(pred)) > 1 else float("nan")
    ari = adjusted_rand_score(labels, pred)
    return sil, ari, pred


def _keep_labels(folios, meta, valid):
    """Оставить только фолио, у которых метка входит в valid (например A/B)."""
    keep = [i for i, f in enumerate(folios) if meta.get(f) in valid]
    return keep


def permutation_test_adjacent(lines_df, section, n_iter=10000, seed=42, unit="quire"):
    """
    Для секции: сравниваем сходство (косинус TF-IDF) между СОСЕДНИМИ единицами
    (unit='quire' для herbal, где много тетрадей; unit='folio' для bio, где вся
    секция в одной тетради и тетради сравнивать нельзя) vs СЛУЧАЙНЫМИ парами.
    Статья: herbal p=0.80, bio p=0.55 (не отличаются).
    """
    rng = np.random.default_rng(seed)
    sec = lines_df[lines_df["section"] == section]

    def folio_order(f):
        import re
        m = re.match(r"f(\d+)([rv])", f)
        return (int(m.group(1)), 0 if m.group(2) == "r" else 1)

    if unit == "quire":
        by_unit = collections.defaultdict(list)
        for toks, quire in zip(sec["tokens"], sec["quire"]):
            by_unit[quire].extend(toks)
        units = sorted([u for u, t in by_unit.items() if len(t) >= 10])
    else:  # folio
        by_unit = collections.defaultdict(list)
        for toks, folio in zip(sec["tokens"], sec["folio"]):
            by_unit[folio].extend(toks)
        units = sorted([u for u, t in by_unit.items() if len(t) >= 5], key=folio_order)

    if len(units) < 3:
        return {"section": section, "unit": unit, "n_units": len(units), "p": float("nan"),
                "real_adj_cosine": float("nan"), "null_mean": float("nan")}

    lists = [by_unit[u] for u in units]
    vocab = common.build_vocab(lists, min_count=1)
    counts = common.count_matrix(lists, vocab)
    X = common.tfidf_matrix(counts)

    def mean_cosine(pairs):
        vals = [common.cosine(X[a], X[b]) for a, b in pairs]
        return float(np.mean(vals)) if vals else 0.0

    adj_pairs = [(i, i + 1) for i in range(len(units) - 1)]
    real = mean_cosine(adj_pairs)

    n = len(units)
    null = np.empty(n_iter)
    for it in range(n_iter):
        idx_a = rng.integers(0, n, size=len(adj_pairs))
        idx_b = rng.integers(0, n, size=len(adj_pairs))
        pairs = [(int(a), int(b)) for a, b in zip(idx_a, idx_b) if a != b]
        null[it] = mean_cosine(pairs[:len(adj_pairs)] or [(0, 1)])
    p = float(np.mean(null >= real))
    return {
        "section": section, "unit": unit, "n_units": len(units),
        "real_adj_cosine": real, "null_mean": float(np.mean(null)),
        "p": p, "n_iter": n_iter,
    }


def run(lines_df, verbose=True) -> dict:
    out = {}
    folios, X, by_folio = folio_vectors(lines_df, min_df=2)

    # собрать метки по фолио
    folio_meta = {}
    for f, row in zip(lines_df["folio"], lines_df[["section", "currier", "scribe"]].itertuples(index=False)):
        folio_meta.setdefault(f, {"section": row.section, "currier": row.currier, "scribe": row.scribe})

    sec_labels = np.array([folio_meta[f]["section"] for f in folios])
    cur_labels = np.array([folio_meta[f]["currier"] for f in folios])
    scr_labels = np.array([folio_meta[f]["scribe"] for f in folios])

    if verbose:
        print("=== ГИПОТЕЗА 3: кластеризация писцы/секции ===")
        print(f"фолио в анализе: {len(folios)}, размерность TF-IDF: {X.shape[1]}")
        print(f"Currier: {collections.Counter(cur_labels)}")
        print(f"section: {collections.Counter(sec_labels)}")
        print(f"scribe:  {collections.Counter(scr_labels)}")
        print()

    # Currier A/B на полном наборе (берём только A/B, не '?')
    keep_ab = np.array([l in ("A", "B") for l in cur_labels])
    Xab = X[keep_ab]
    cur_ab = cur_labels[keep_ab]
    cur_ab_enc = np.array([0 if l == "A" else 1 for l in cur_ab])
    # на сыром TF-IDF (как в статье: silhouette по исходным признакам)
    sil_cur_raw, ari_cur_raw, _ = cluster_eval(Xab, cur_ab_enc, n_clusters=2, use_umap=False)
    # на UMAP-редукции
    sil_cur_umap, ari_cur_umap, _ = cluster_eval(Xab, cur_ab_enc, n_clusters=2, use_umap=True)
    out["currier_AB"] = {"silhouette_raw": float(sil_cur_raw), "ari_2cl_raw": float(ari_cur_raw),
                         "silhouette_umap": float(sil_cur_umap), "ari_2cl_umap": float(ari_cur_umap),
                         "n_folios": int(keep_ab.sum())}
    if verbose:
        print(f"Currier A/B (2 кластера):")
        print(f"  silhouette на TF-IDF: {sil_cur_raw:+.3f}, ARI={ari_cur_raw:.3f}")
        print(f"  silhouette на UMAP:   {sil_cur_umap:+.3f}, ARI={ari_cur_umap:.3f}")
        print(f"  (статья: silhouette +0.15, ARI до 0.64)")
        print()

    # section & scribe: silhouette (K-means vs меток). Используем silhouette самой кластеризации
    # и ARI. Для 'отрицательного silhouette' сравниваем с реальной кластеризацией.
    for name, labels in [("section", sec_labels), ("scribe", scr_labels)]:
        # берём только часто встречающиеся метки
        cnt = collections.Counter(labels)
        valid = {k for k, v in cnt.items() if v >= 3 and k != "?"}
        keep = np.array([l in valid for l in labels])
        Xs = X[keep]; ls = labels[keep]
        n_cl = len(set(ls))
        if n_cl < 2 or Xs.shape[0] < n_cl * 2:
            out[name] = {"silhouette_raw": float("nan"), "ari_raw": float("nan")}
            continue
        sil, ari, _ = cluster_eval(Xs, ls, n_clusters=n_cl, use_umap=False)
        out[name] = {"silhouette_raw": float(sil), "ari_raw": float(ari), "n_clusters": n_cl}
        if verbose:
            print(f"{name} (KMeans k={n_cl}, TF-IDF): silhouette={sil:+.3f}, ARI={ari:.3f}")
    if verbose:
        print(f"  (статья: section и scribe - отрицательный silhouette, хуже случайного)")
        print()

    # внутри Currier B: bio vs остальное
    keepB = np.array([l == "B" for l in cur_labels])
    XB = X[keepB]
    secB = sec_labels[keepB]
    cntB = collections.Counter(secB)
    validB = {k for k, v in cntB.items() if v >= 3}
    keepB2 = np.array([l in validB for l in secB])
    XB2 = XB[keepB2]; secB2 = secB[keepB2]
    if len(set(secB2)) >= 2 and XB2.shape[0] >= 4:
        # bio vs rest -> 2 кластера
        bio_enc = np.array([0 if l == "bio" else 1 for l in secB2])
        silB, ariB, _ = cluster_eval(XB2, bio_enc, n_clusters=2, use_umap=False)
        out["within_B_bio"] = {"silhouette": float(silB), "ari": float(ariB), "n": int(len(secB2))}
        if verbose:
            print(f"Внутри Currier B (bio vs rest, 2 кластера, TF-IDF): silhouette={silB:+.3f}, ARI={ariB:.3f}")
            print(f"  (статья: внутри B bio ARI ~0.25)")
    else:
        if verbose:
            print("Внутри Currier B: недостаточно данных")
    if verbose:
        print()

    # тест перестановок соседних единиц
    for section, unit in [("herbal", "quire"), ("bio", "folio")]:
        r = permutation_test_adjacent(lines_df, section, n_iter=2000, unit=unit)
        out[f"perm_{section}"] = r
        if verbose:
            print(f"Тест перестановок соседних {unit} [{section}]: "
                  f"real cos={r.get('real_adj_cosine'):.3f}, null={r.get('null_mean'):.3f}, "
                  f"p={r.get('p'):.2f}, n_{unit}={r.get('n_units')}")
    if verbose:
        print(f"  (статья: herbal p=0.80, bio p=0.55 - не отличаются)")
        print()
    return out


if __name__ == "__main__":
    from voi.parse_ivtff import parse_ivtff, add_n_token_columns
    ld, _ = parse_ivtff(str(common.data_path("voynich")))
    ld = add_n_token_columns(ld)
    run(ld)
