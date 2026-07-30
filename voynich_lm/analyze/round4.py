"""
Раунд 4: шесть новых подходов к языковой структуре Войнича.

A. Закон Ципфа + Heaps law      — классический тест «язык/не язык»
B. Mutual Information по дистанции — long-range correlations
C. Марков k=0..3 vs Трансформер  — локальная/нелокальная структура
D. Позиционный профиль глифов    — фонетика (где стоят классы глифов)
E. PPMI+SVD семантические векторы — тест смысла через word-embeddings
F. Temporal drift                — эволюция статистики по рукописи

Каждый сравнивает Войнич с теми же 4 контрольными текстами + shuffle-null.
Все анализы deterministic, на CPU, быстрые.
"""
from __future__ import annotations
import json
import math
import re
import collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, ".")
from voi.parse_ivtff import parse_ivtff, add_n_token_columns
from voi.common import data_path


# ============================================================
# Подготовка текстов: word-tokens для Войнича и контролов
# ============================================================

def voynich_words() -> list[str]:
    ld, _ = parse_ivtff(str(data_path("voynich")))
    ld = add_n_token_columns(ld)
    txt = ld[ld["n_tokens"] > 0]
    return [t for ts in txt["tokens"] for t in ts]


def control_words(name: str, max_words: int = 31000) -> list[str]:
    """Word-tokens контрольного текста (обрезано до длины Войнича)."""
    raw = Path(str(data_path(name))).read_text(encoding="utf-8", errors="ignore")
    s = raw.find("*** START OF"); e = raw.find("*** END OF")
    if s != -1 and e != -1:
        body = raw[s:e]
        nl = body.find("\n")
        if nl != -1: body = body[nl+1:]
    else:
        body = raw
    words = re.findall(r"[a-z]+", body.lower())
    return words[:max_words]


def shuffle_words(words: list[str], seed: int = 7) -> list[str]:
    rng = np.random.default_rng(seed)
    w = list(words); rng.shuffle(w)
    return w


# ============================================================
# A. Закон Ципфа + Heaps
# ============================================================

def zipf_heaps(words: list[str]) -> dict:
    counts = collections.Counter(words)
    rank = sorted(counts.values(), reverse=True)
    n = len(words)
    V = len(counts)
    # Heaps: V как функция n (по приращениям)
    seen = set(); heap_x = []; heap_y = []
    for i, w in enumerate(words):
        seen.add(w)
        if i % 500 == 0 or i == n-1:
            heap_x.append(i+1); heap_y.append(len(seen))
    return {"zipf_ranks": rank[:100], "n_tokens": n, "n_types": V,
            "heaps_n": heap_x, "heaps_v": heap_y,
            "zipf_slope": _zipf_slope(rank)}


def _zipf_slope(ranks: list[int]) -> float:
    """Наклон log(f) vs log(rank) для топ-100. У естественных языков ~ -1.0."""
    top = ranks[:100]
    logr = np.log10(np.arange(1, len(top)+1))
    logf = np.log10(np.array(top, dtype=float))
    if len(top) < 5: return float("nan")
    return float(np.polyfit(logr, logf, 1)[0])


# ============================================================
# B. Mutual Information по дистанции
# ============================================================

def mutual_info_by_distance(words: list[str], max_dist: int = 32) -> np.ndarray:
    """
    I(w_i ; w_{i+d}) как функция d. Long-range correlation.
    В естественных языках MI медленно убывает (тематическая связность);
    в случайном потоке — быстро падает к 0 после d=1-2.
    """
    # бинаризуем по частому/редкому, чтобы MI было вычислимо на малом корпусе
    counts = collections.Counter(words)
    # топ-100 частых как «события»
    frequent = {w for w, c in counts.most_common(100)}
    idx = [w if w in frequent else "<UNK>" for w in words]
    mi = np.zeros(max_dist)
    for d in range(1, max_dist+1):
        # совместная и маргинальные частоты
        joint = collections.Counter()
        px = collections.Counter()
        py = collections.Counter()
        n = 0
        for i in range(len(idx) - d):
            a, b = idx[i], idx[i+d]
            joint[(a,b)] += 1; px[a] += 1; py[b] += 1
            n += 1
        if n == 0: continue
        total_mi = 0.0
        for (a,b), c in joint.items():
            pxy = c/n; pax = px[a]/n; pby = py[b]/n
            if pxy > 0 and pax > 0 and pby > 0:
                total_mi += pxy * math.log2(pxy/(pax*pby))
        mi[d-1] = total_mi
    return mi


# ============================================================
# C. Марков k=0..3 vs оценка трансформера
# ============================================================

def markov_entropy_estimates(words: list[str]) -> dict:
    """
    H_0 (униграммная), H_1, H_2, H_3 энтропия по словам (бит/слово).
    Сравнение: насколько каждый порядок улучшает предсказание.
    У «языка» резкое падение H_0->H_1->H_2 (контекст помогает); у имитации
    с локальной структурой падение концентрируется на H_1 и быстро насыщается.
    """
    def entropy_ngram(tokens, n):
        if n == 1:
            c = collections.Counter(tokens)
            tot = sum(c.values()); return -sum((v/tot)*math.log2(v/tot) for v in c.values())
        ng = collections.Counter()
        for i in range(len(tokens)-n+1):
            ng[tuple(tokens[i:i+n])] += 1
        # условная H = H(ngram) - H(n-1gram) слишком грубо на малом корпусе;
        # вместо этого —Leave-one-out-style: средняя -log2 P(w_i | prefix)
        # через сглаженные частоты
        prefix_counts = collections.Counter()
        ng_counts = collections.Counter()
        for i in range(len(tokens)-n+1):
            ng_counts[tuple(tokens[i:i+n])] += 1
            prefix_counts[tuple(tokens[i:i+n-1])] += 1
        # средняя условная энтропия с add-1
        import numpy as np
        V = len(set(tokens))
        ll = 0.0; cnt = 0
        for i in range(len(tokens)-n+1):
            prefix = tuple(tokens[i:i+n-1])
            w = tokens[i+n-1]
            num = ng_counts[prefix+(w,)] + 0.1
            den = prefix_counts[prefix] + 0.1*V
            ll += -math.log2(num/den); cnt += 1
        return ll/cnt if cnt else float("nan")

    out = {}
    for k in range(4):
        out[f"H_{k}"] = entropy_ngram(words, k+1)
    return out


# ============================================================
# D. Позиционный профиль глифов
# ============================================================

def positional_glyph_profile(words: list[str], top_glyphs: list[str]) -> dict:
    """
    Распределение каждого глифа по норм. позиции в слове [0..1].
    Гласные в естественных языках тяготеют к середине/определённым позициям,
    аффрикаты/маркеры — к краям. Сравним форму распределений.
    """
    pos = {g: [] for g in top_glyphs}
    for w in words:
        L = len(w)
        if L < 2: continue
        for i, c in enumerate(w):
            if c in pos:
                pos[c].append(i / (L-1))
    # гистограмма по 5 бинам
    bins = np.linspace(0, 1, 6)
    profiles = {}
    for g, ps in pos.items():
        if len(ps) < 50: continue
        h, _ = np.histogram(ps, bins=bins)
        profiles[g] = (h / h.sum()).tolist()
    return profiles


# ============================================================
# E. PPMI+SVD семантические векторы
# ============================================================

def ppmi_svd_embeddings(words: list[str], window: int = 5, dim: int = 50,
                         min_count: int = 5, top_vocab: int = 300) -> tuple:
    """
    PPMI (positive PMI) матрица ко-встреч + SVD -> семантические векторы слов.
    Если в тексте есть смысл, семантически близкие слова кластеризуются
    (синонимы/тематически связанные). Тест: устойчивость кластеров vs shuffle.
    """
    counts = collections.Counter(words)
    vocab = [w for w, c in counts.most_common(top_vocab) if c >= min_count]
    vidx = {w: i for i, w in enumerate(vocab)}
    # контекстная матрица ко-встреч
    M = np.zeros((len(vocab), len(vocab)))
    for i in range(len(words)):
        if words[i] not in vidx: continue
        for d in range(1, window+1):
            for j in (i-d, i+d):
                if 0 <= j < len(words) and words[j] in vidx:
                    M[vidx[words[i]], vidx[words[j]]] += 1
    # PPMI
    total = M.sum()
    if total == 0: return None, vocab
    row = M.sum(1); col = M.sum(0)
    PPMI = np.zeros_like(M)
    for i in range(len(vocab)):
        for j in range(len(vocab)):
            if M[i,j] > 0 and row[i] > 0 and col[j] > 0:
                pmi = math.log2((M[i,j]*total) / (row[i]*col[j]))
                PPMI[i,j] = max(pmi, 0)
    # SVD
    U, S, Vt = np.linalg.svd(PPMI, full_matrices=False)
    emb = U[:, :dim] * np.sqrt(S[:dim])
    return emb, vocab


def embedding_cluster_quality(emb: np.ndarray, vocab: list[str], n_clusters: int = 8) -> dict:
    """Кластеризуем word-embeddings и мерим silhouette. Сравнение vs shuffle."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    if emb is None or len(vocab) < n_clusters*2: return {"silhouette": float("nan")}
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=7)
    labels = km.fit_predict(emb)
    sil = silhouette_score(emb, labels) if len(set(labels)) > 1 else float("nan")
    return {"silhouette": float(sil), "n_words": len(vocab)}


# ============================================================
# F. Temporal drift по рукописи
# ============================================================

def temporal_drift() -> dict:
    """
    Эволюция статистики вдоль рукописи (по фолио в порядке folio_num):
    TTR, частота топ-глифов, длина слова. Есть ли резкие change-points?
    """
    ld, _ = parse_ivtff(str(data_path("voynich")))
    ld = add_n_token_columns(ld)
    ld = ld.sort_values(["folio_num", "locus_seq"]).reset_index(drop=True)
    # по фолио: агрегаты
    by_folio = []
    for f, grp in ld.groupby("folio"):
        toks = [t for ts in grp["tokens"] if ts for t in ts]
        if not toks: continue
        by_folio.append({
            "folio": f, "folio_num": int(grp["folio_num"].iloc[0]),
            "section": grp["section"].iloc[0], "currier": grp["currier"].iloc[0],
            "ttr": len(set(toks))/len(toks),
            "mean_wlen": np.mean([len(t) for t in toks]),
            "ntok": len(toks),
        })
    by_folio.sort(key=lambda x: x["folio_num"])
    return {"folios": by_folio}


# ============================================================
# Оркестр
# ============================================================

def run(figures: Path = Path("figures")) -> dict:
    figures.mkdir(exist_ok=True)
    print("=" * 68)
    print("РАУНД 4: шесть новых подходов к структуре Войнича")
    print("=" * 68)

    texts = {
        "voynich": voynich_words(),
        "kjv": control_words("kjv"),
        "moby": control_words("moby"),
        "paradise": control_words("paradise"),
        "caesar": control_words("caesar"),
    }
    texts["shuffle"] = shuffle_words(texts["voynich"])
    results = {}

    # ---- A. Ципф + Heaps ----
    print("\n--- A. Закон Ципфа + Heaps ---")
    print(f"  {'текст':10s} {'токены':7s} {'типы':6s} {'наклон Ципфа':>14s}")
    zipf_data = {}
    for name, w in texts.items():
        z = zipf_heaps(w)
        zipf_data[name] = z
        print(f"  {name:10s} {z['n_tokens']:7d} {z['n_types']:6d} {z['zipf_slope']:14.3f}")
    results["zipf_heaps"] = {n: {"n_tokens": z["n_tokens"], "n_types": z["n_types"],
                                  "zipf_slope": z["zipf_slope"]} for n, z in zipf_data.items()}
    # график
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = {"voynich":"#c0392b","kjv":"#2980b9","moby":"#27ae60",
              "paradise":"#8e44ad","caesar":"#e67e22","shuffle":"#7f8c8d"}
    for name, z in zipf_data.items():
        axes[0].plot(range(1, len(z["zipf_ranks"])+1), z["zipf_ranks"],
                     colors[name], label=name, lw=1.5 if name=="voynich" else 1, alpha=0.85)
        axes[1].plot(z["heaps_n"], z["heaps_v"], colors[name],
                     label=name, lw=1.5 if name=="voynich" else 1, alpha=0.85)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_title("Закон Ципфа (log-log)"); axes[0].set_xlabel("ранг"); axes[0].set_ylabel("частота")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_title("Heaps law (рост словаря)"); axes[1].set_xlabel("токены N"); axes[1].set_ylabel("типы V(N)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figures / "R4_A_zipf_heaps.png", dpi=130)
    print(f"  -> figures/R4_A_zipf_heaps.png")

    # ---- B. MI по дистанции ----
    print("\n--- B. Mutual Information по дистанции ---")
    print(f"  {'текст':10s} {'MI@1':>8s} {'MI@2':>8s} {'MI@5':>8s} {'MI@10':>8s} {'MI@20':>8s}")
    mi_data = {}
    for name, w in texts.items():
        mi = mutual_info_by_distance(w, max_dist=32)
        mi_data[name] = mi.tolist()
        idx = [0,1,4,9,19]
        cells = "  ".join(f"{mi[i]:8.3f}" for i in idx)
        print(f"  {name:10s} {cells}")
    results["mi_by_distance"] = mi_data
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, mi in mi_data.items():
        ax.plot(range(1, len(mi)+1), mi, "o-", color=colors[name], label=name,
                lw=1.8 if name=="voynich" else 1, alpha=0.85, ms=3)
    ax.set_xlabel("дистанция d (слова)"); ax.set_ylabel("Mutual Information (бит)")
    ax.set_title("Long-range correlations: MI(w_i, w_{i+d})"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figures / "R4_B_mutual_info.png", dpi=130)
    print(f"  -> figures/R4_B_mutual_info.png")

    # ---- C. Марков ----
    print("\n--- C. Марков k=0..3 (бит/слово) ---")
    print(f"  {'текст':10s} {'H_0':>8s} {'H_1':>8s} {'H_2':>8s} {'H_3':>8s} {'ΔH1-0':>8s} {'ΔH2-1':>8s}")
    markov_data = {}
    for name, w in texts.items():
        m = markov_entropy_estimates(w)
        markov_data[name] = m
        d10 = m["H_0"]-m["H_1"]; d21 = m["H_1"]-m["H_2"]
        print(f"  {name:10s} {m['H_0']:8.2f} {m['H_1']:8.2f} {m['H_2']:8.2f} {m['H_3']:8.2f} {d10:8.2f} {d21:8.2f}")
    results["markov_entropy"] = markov_data
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(4)
    for name, m in markov_data.items():
        ax.plot(x, [m[f"H_{k}"] for k in range(4)], "o-", color=colors[name],
                label=name, lw=1.8 if name=="voynich" else 1, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(["H_0 (unigram)","H_1","H_2","H_3"])
    ax.set_ylabel("условная энтропия (бит/слово)")
    ax.set_title("Насыщение предсказания: уни→би→три\n(резкое насыщение = имитация с локальной структурой)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figures / "R4_C_markov.png", dpi=130)
    print(f"  -> figures/R4_C_markov.png")

    # ---- D. Позиционный профиль ----
    print("\n--- D. Позиционный профиль глифов Войнича ---")
    top_glyphs = list("oehyacdikl")
    prof = positional_glyph_profile(texts["voynich"], top_glyphs)
    results["positional_profile_voynich"] = prof
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bin_centers = np.linspace(0.1, 0.9, 5)
    cmap = plt.cm.tab10
    for i, (g, h) in enumerate(prof.items()):
        ax.plot(bin_centers, h, "o-", color=cmap(i/10), label=g, lw=1.5)
    ax.set_xlabel("норм. позиция в слове (0=начало, 1=конец)")
    ax.set_ylabel("доля встречаемости глифа")
    ax.set_title("Позиционный профиль глифов Войнича\n(где в слове стоит каждый глиф)")
    ax.legend(ncol=5); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figures / "R4_D_position.png", dpi=130)
    print(f"  -> figures/R4_D_position.png")

    # ---- E. PPMI+SVD ----
    print("\n--- E. PPMI+SVD семантические векторы (silhouette кластеров) ---")
    print(f"  {'текст':10s} {'silhouette':>12s}")
    ppmi_data = {}
    for name, w in texts.items():
        emb, vocab = ppmi_svd_embeddings(w)
        q = embedding_cluster_quality(emb, vocab)
        ppmi_data[name] = q
        s = q["silhouette"]
        print(f"  {name:10s} {s:12.3f}" if s==s else f"  {name:10s}          n/a")
    results["ppmi_clusters"] = ppmi_data
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = list(ppmi_data.keys()); sils = [ppmi_data[n]["silhouette"] for n in names]
    ax.bar(names, [s if s==s else 0 for s in sils],
           color=[colors[n] for n in names])
    ax.set_ylabel("silhouette (PPMI+SVD word clusters)")
    ax.set_title("Качество семантических кластеров слов\n(есть ли структура смысла?)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(figures / "R4_E_ppmi.png", dpi=130)
    print(f"  -> figures/R4_E_ppmi.png")

    # ---- F. Temporal drift ----
    print("\n--- F. Temporal drift по рукописи ---")
    drift = temporal_drift()
    results["temporal_drift"] = {"n_folios": len(drift["folios"])}
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    folios = drift["folios"]
    fnums = [f["folio_num"] for f in folios]
    ttr = [f["ttr"] for f in folios]
    wlen = [f["mean_wlen"] for f in folios]
    # цвет по currier
    cur_colors = {"A":"#2980b9","B":"#c0392b","?":"#7f8c8d"}
    cc = [cur_colors.get(f["currier"],"#999") for f in folios]
    axes[0].scatter(fnums, ttr, c=cc, s=14, alpha=0.7)
    axes[0].set_ylabel("TTR"); axes[0].set_title("Дрейф статистики по рукописи (синий=Currier A, красный=B)")
    axes[1].scatter(fnums, wlen, c=cc, s=14, alpha=0.7); axes[1].set_ylabel("средняя длина слова")
    axes[2].scatter(fnums, [f["ntok"] for f in folios], c=cc, s=14, alpha=0.7)
    axes[2].set_ylabel("токенов на фолио"); axes[2].set_xlabel("номер фолио")
    for ax in axes: ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figures / "R4_F_temporal.png", dpi=130)
    print(f"  -> figures/R4_F_temporal.png")

    Path("round4_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  -> round4_results.json")
    return results


if __name__ == "__main__":
    run()
