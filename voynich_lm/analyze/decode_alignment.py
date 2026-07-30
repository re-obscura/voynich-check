"""
Направление 3: Procrustes-выравнивание эмбеддингов -> фонетические гипотезы.

Идея: glyph-эмбеддинги Войнича и char-эмбеддинги латиницы (из модели KJV)
находятся в пространствах одинаковой размерности (d=128), но разных системах
координат. Procrustes (ортогональное выравнивание) сажает «поведенчески похожие»
глифы рядом с буквами — по контексту, без угадывания языка.

Внимание: это ГЕНЕРАТОР ГИПОТЕЗ, а не расшифровка. Результаты — кандидаты на
звуки/классы, которые скармливаются направлению 1 (LM-перебор шифра) для проверки.

Также: классификация глифов Войнича на гласные/согласные через их поведение
(гласные в естественных языках: высокая частота, равномерное распределение по
позициям; согласные — более избирательны).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, ".")
from voynich_lm.train import load_model


def glyph_emb_matrix(name: str, device=None) -> tuple[np.ndarray, list[str]]:
    """Матрица эмбеддингов (n, d) и список токенов (только реальные глифы, без спец)."""
    model, tok = load_model(name, device=device)
    W = model.tok_emb.weight.detach().cpu().numpy()
    # реальные глифы — не спецтокены (<...>)
    idx = [i for i, t in enumerate(tok.itos) if not (t.startswith("<") or t == "")]
    return W[idx], [tok.itos[i] for i in idx]


def procrustes(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Ортогонанальное выравнивание A -> B. Возвращает (R, aligned_A, frob_error).
    A, B: (n, d), n может различаться — выравниваем общие строки по размеру меньшего.
    """
    n = min(A.shape[0], B.shape[0])
    A2, B2 = A[:n].copy(), B[:n].copy()
    # центрируем
    A2 -= A2.mean(0); B2 -= B2.mean(0)
    # нормируем
    A2 /= (np.linalg.norm(A2) + 1e-9); B2 /= (np.linalg.norm(B2) + 1e-9)
    M = A2.T @ B2
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    aligned = A2 @ R
    err = np.linalg.norm(aligned - B2)
    return R, aligned, err


def nearest_neighbors_aligned(aligned_voyn: np.ndarray, voyn_glyphs: list[str],
                                latin_emb: np.ndarray, latin_chars: list[str],
                                topk: int = 3) -> dict:
    """Для каждого войничского глифа — top-k ближайших латинских букв (косинус)."""
    def normalize(X):
        n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
        return X / n
    Vn = normalize(aligned_voyn)
    Ln = normalize(latin_emb)
    out = {}
    for i, g in enumerate(voyn_glyphs):
        sims = Vn[i] @ Ln.T
        top = np.argsort(-sims)[:topk]
        out[g] = [(latin_chars[j], float(sims[j])) for j in top]
    return out


def vowel_consonant_classification(lines_df=None) -> dict:
    """
    Классификация глифов Войнича на гласные/согласные по поведению:
      гласные: высокая частота, низкая условная энтропия (равномерны по соседям),
               тяготеют к центрам слов.
    Используем корпусную статистику (как в компаративной лингвистике).
    """
    from voi.parse_ivtff import parse_ivtff, add_n_token_columns
    from voi.common import data_path
    import collections, math
    if lines_df is None:
        ld, _ = parse_ivtff(str(data_path("voynich")))
        ld = add_n_token_columns(ld)
        lines_df = ld[ld["n_tokens"] > 0]
    alltok = [t for ts in lines_df["tokens"] if ts for t in ts]
    # частота глифов
    freq = collections.Counter()
    for t in alltok:
        freq.update(t)
    total = sum(freq.values())
    fr = {c: freq[c] / total for c in freq}
    # позиция в слове: гласные чаще в середине, согласные на краях
    pos_score = {c: [] for c in freq}  # норм. позиция глифа в слове [0..1]
    for t in alltok:
        L = len(t)
        for i, c in enumerate(t):
            pos_score[c].append(i / max(L - 1, 1))
    mid_tendency = {c: float(np.mean(pos_score[c])) for c in freq}  # ~0.5 = середина
    # биграммная энтропия соседей (гласные = более разнообразные соседи)
    # H(сосед | глиф)
    left = collections.defaultdict(collections.Counter)
    right = collections.defaultdict(collections.Counter)
    for t in alltok:
        for i, c in enumerate(t):
            if i > 0:
                left[c][t[i-1]] += 1
            if i < len(t) - 1:
                right[c][t[i+1]] += 1
    def H(counter):
        n = sum(counter.values()); 
        if n == 0: return 0.0
        return -sum((v/n) * math.log2(v/n) for v in counter.values())
    neigh_entropy = {c: (H(left[c]) + H(right[c])) / 2 for c in freq}

    # комбинированный «vowelness score»: высокая частота + середина + разнообразные соседи
    chars = sorted(freq.keys())
    fr_arr = np.array([fr[c] for c in chars])
    mid_arr = np.array([mid_tendency[c] for c in chars])
    nh_arr = np.array([neigh_entropy[c] for c in chars])
    # z-score нормировка
    def z(x): return (x - x.mean()) / (x.std() + 1e-9)
    vowelness = z(np.log(fr_arr + 1e-9)) - z(np.abs(mid_arr - 0.5)) + z(nh_arr)
    order = np.argsort(-vowelness)
    ranked = [(chars[i], float(vowelness[i]), fr[chars[i]],
               mid_tendency[chars[i]], neigh_entropy[chars[i]]) for i in order]
    # эвристический порог: топ-5 — кандидаты в гласные
    vowels_candidates = [c for c, *_ in ranked[:5]]
    return {"ranked": ranked, "vowel_candidates": vowels_candidates,
            "freq": fr, "mid_tendency": mid_tendency, "neigh_entropy": neigh_entropy}


def run(figures: Path = Path("figures"), out: dict | None = None) -> dict:
    figures.mkdir(exist_ok=True)
    print("=" * 68)
    print("НАПРАВЛЕНИЕ 3: Procrustes-выравнивание эмбеддингов")
    print("=" * 68)

    Wv, gv = glyph_emb_matrix("voynich")
    Wk, gk = glyph_emb_matrix("kjv")   # латиница: a-z + пробел
    print(f"\n  Войнич: {len(gv)} глифов, KJV: {len(gk)} символов латиницы")

    # Procrustes: выравниваем Войнич в систему KJV
    R, aligned, err = procrustes(Wv, Wk)
    print(f"  Procrustes frobenius error: {err:.4f} (меньше = лучше выравнивание)")

    # nearest neighbors
    nn = nearest_neighbors_aligned(aligned, gv, Wk, gk, topk=3)
    print(f"\n  Ближайшие латинские буквы для каждого войничского глифа (по контексту):")
    print(f"  {'глиф':6s} {'top-3 латинских':30s}")
    for g in gv:
        tps = " ".join(f"{c}({s:.2f})" for c, s in nn[g])
        print(f"  {g:6s} {tps}")

    # классификация гласные/согласные
    print(f"\n  Классификация глифов Войнича на гласные/согласные (по поведению):")
    print(f"  (частота, стремление к середине слова, разнообразие соседей)")
    vc = vowel_consonant_classification()
    print(f"  {'глиф':6s} {'vowelness':>10s} {'freq':>7s} {'mid':>5s} {'H(сосед)':>9s}")
    for c, v, fr, mid, nh in vc["ranked"]:
        marker = "  <- vowel?" if c in vc["vowel_candidates"] else ""
        print(f"  {c:6s} {v:10.2f} {fr:7.3f} {mid:5.2f} {nh:9.2f}{marker}")
    print(f"\n  Кандидаты в гласные (топ-5): {vc['vowel_candidates']}")

    # --- график: 2D проекция выровненных эмбеддингов ---
    from sklearn.decomposition import PCA
    fig, ax = plt.subplots(figsize=(10, 8))
    # PCA по объединённой матрице для общей плоскости
    combined = np.vstack([aligned, Wk])
    # выровним Wk тождественно (он уже в своей системе)
    pca = PCA(n_components=2)
    P = pca.fit_transform(combined)
    Pv, Pk = P[:len(gv)], P[len(gv):]
    ax.scatter(Pv[:, 0], Pv[:, 1], s=120, c="#c0392b", zorder=3, label="Voynich glyphs")
    ax.scatter(Pk[:, 0], Pk[:, 1], s=120, c="#2980b9", zorder=3, marker="s", label="Latin (KJV)")
    for i, g in enumerate(gv):
        ax.annotate(g, (Pv[i, 0], Pv[i, 1]), fontsize=11, fontweight="bold",
                    color="#c0392b", xytext=(5, 5), textcoords="offset points")
    for i, c in enumerate(gk):
        ax.annotate(c, (Pk[i, 0], Pk[i, 1]), fontsize=10, color="#2980b9",
                    xytext=(5, 5), textcoords="offset points")
    ax.set_title("Procrustes-выравнивание: глифы Войнича ↔ латиница (PCA 2D)\n"
                 "Близкие точки = поведенчески похожие (по контексту)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "DEC_alignment.png", dpi=130)
    print(f"\n  -> {figures/'DEC_alignment.png'}")

    # сохраняем гипотезы для направления 1
    hyps = {g: nn[g] for g in gv}
    Path("decode_hypotheses.json").write_text(
        __import__("json").dumps({"procrustes_nn": hyps,
                                  "vowel_candidates": vc["vowel_candidates"],
                                  "vowel_consonant_ranked": [(c, v) for c, v, *_ in vc["ranked"]]},
                                 indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> decode_hypotheses.json (для направления 1)")

    if out is not None:
        out["alignment"] = {"procrustes_error": err, "nearest_neighbors": hyps,
                            "vowel_candidates": vc["vowel_candidates"]}
    return {"procrustes_error": err, "nearest_neighbors": hyps,
            "vowel_candidates": vc["vowel_candidates"]}


if __name__ == "__main__":
    run()
