"""
G: эмбеддинги глифов/слов + интерактивная 3D-карта памяти.

Извлекаем выученные эмбеддингли глифов для каждого из 6 языков. Сводим в общее
пространство через простой трюк: т.к. словари разные (EVA vs латиница), сравниваем
не самих «соседей», а ГЕОМЕТРИЮ — распределение попарных косинусных расстояний
(«форму» пространства памяти). Для визуализации строим отдельные UMAP-проекции
для каждого языка и общую таблицу статистик.

Дополнительно: для Войнича — эмбеддинги слов (среднее глиф-эмбеддингов),
кластеризуем, смотрим интерпретируемость (префиксы/суффиксы).

Интерактивная 3D-карта: plotly -> вращаемый .html.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from voynich_lm.train import load_model


def glyph_embeddings(name: str, device: str | None = None) -> tuple[np.ndarray, list[str]]:
    """Матрица эмбеддингов глифов (n_glyphs, d) и список токенов словаря."""
    model, tok = load_model(name, device=device)
    W = model.tok_emb.weight.detach().cpu().numpy()
    return W, tok.itos


def pairwise_distance_stats(W: np.ndarray) -> dict:
    """Статистика геометрии пространства: распределение попарных косинусных расстояний."""
    # нормируем
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms == 0] = 1
    Wn = W / norms
    cos = Wn @ Wn.T
    iu = np.triu_indices(cos.shape[0], k=1)
    dists = 1 - cos[iu]
    return {"mean": float(np.mean(dists)), "std": float(np.std(dists)),
            "median": float(np.median(dists)), "min": float(np.min(dists)),
            "max": float(np.max(dists)), "n": int(len(dists))}


def word_embeddings_voynich(device: str | None = None, min_count: int = 10) -> tuple[np.ndarray, list[str]]:
    """Эмбеддинги слов Войнича = среднее глиф-эмбеддингов. Только частые слова."""
    from voynich_lm.data import build_voynich_dataset
    from voi.parse_ivtff import parse_ivtff, add_n_token_columns
    from voi.common import data_path
    import collections
    ds = build_voynich_dataset()
    model, tok = load_model("voynich", device=device)
    W = model.tok_emb.weight.detach().cpu().numpy()
    # частоты слов
    ld, _ = parse_ivtff(str(data_path("voynich")))
    ld = add_n_token_columns(ld)
    toks_all = [t for ts in ld["tokens"] if ts for t in ts]
    freq = collections.Counter(toks_all)
    words = [w for w, c in freq.items() if c >= min_count]
    vecs = []
    for w in words:
        idxs = [tok.stoi.get(ch) for ch in w if tok.stoi.get(ch) is not None]
        if not idxs:
            continue
        vecs.append(W[idxs].mean(axis=0))
    return np.stack(vecs), words


def run(device: str | None = None,
        figures: Path = Path("figures"), out: dict | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    figures.mkdir(exist_ok=True)

    print("=== G: эмбеддинги + 3D-карта памяти ===")
    results = {}
    embeddings = {}
    vocabs = {}
    for name in ["voynich", "kjv", "moby", "paradise", "caesar", "shuffle"]:
        try:
            W, itos = glyph_embeddings(name, device=device)
            embeddings[name] = W
            vocabs[name] = itos
            st = pairwise_distance_stats(W)
            results[name] = {"vocab_size": len(itos), "d": int(W.shape[1]),
                             "geom_stats": st}
            print(f"  {name:10s}: vocab={len(itos):3d} d={W.shape[1]} "
                  f"pairwise-dist mean={st['mean']:.3f} std={st['std']:.3f}")
        except FileNotFoundError:
            print(f"  {name:10s}: модель не обучена — пропуск")

    # --- 3D-карта памяти (plotly): один UMAP на язык, точки=глифы/буквы ---
    _plot_3d_memory(embeddings, vocabs, figures)
    # --- слова Войнича: UMAP + кластеры ---
    _plot_voynich_words(device, figures)

    if out is not None:
        out["embeddings"] = results
    return results


def _plot_3d_memory(embeddings, vocabs, figures):
    """Интерактивная вращаемая 3D-карта. Для каждого языка — свой UMAP-3D."""
    try:
        import umap
        import plotly.graph_objects as go
    except ImportError:
        print("  plotly/umap недоступны — 3D-карта пропущена")
        return

    # цвета языков
    colors = {"voynich": "#c0392b", "kjv": "#2980b9", "moby": "#27ae60",
              "paradise": "#8e44ad", "caesar": "#e67e22", "shuffle": "#7f8c8d"}

    fig = go.Figure()
    for name, W in embeddings.items():
        itos = vocabs[name]
        # пропустим спецтокены <...> из визуализации
        keep_idx = [i for i, t in enumerate(itos) if not (t.startswith("<") or t == "")]
        if len(keep_idx) < 5:
            continue
        Wk = W[keep_idx]
        labels = [itos[i] for i in keep_idx]
        try:
            reducer = umap.UMAP(n_components=3, n_neighbors=min(8, len(keep_idx)-1),
                                metric="cosine", random_state=7)
            P = reducer.fit_transform(Wk)
        except Exception as e:
            print(f"    UMAP для {name} не удался: {e}")
            continue
        fig.add_trace(go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2],
            mode="markers+text",
            text=labels, textposition="top center", textfont=dict(size=9),
            marker=dict(size=4, color=colors.get(name, "#333"), opacity=0.8),
            name=name,
        ))
    fig.update_layout(
        title="3D-карта памяти: эмбеддинги глифов по языкам (крутите мышкой)",
        scene=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3"),
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, t=40, b=0),
        height=750,
    )
    html_path = figures / "G_3d_memory.html"
    fig.write_html(str(html_path))
    print(f"  -> {html_path}  (откройте в браузере, чтобы покрутить)")
    # статичный снимок
    try:
        fig.write_image(str(figures / "G_3d_memory.png"), scale=1.5)
        print(f"  -> {figures/'G_3d_memory.png'}")
    except Exception:
        # kaleido может отсутствовать — это ОК
        pass


def _plot_voynich_words(device, figures):
    """UMAP эмбеддингов слов Войнича + окраска по префиксу/суффиксу."""
    try:
        import umap
    except ImportError:
        return
    try:
        W, words = word_embeddings_voynich(device=device, min_count=10)
    except FileNotFoundError:
        print("  voynich-модель не обучена — карта слов пропущена")
        return
    if len(words) < 10:
        return
    reducer = umap.UMAP(n_components=2, n_neighbors=15, metric="cosine", random_state=7)
    P = reducer.fit_transform(W)
    # окраска по последним 2 глифам (суффикс) — из H1 мы знаем суффиксы значимы
    def suffix(w):
        return w[-2:] if len(w) >= 2 else w
    sufs = [suffix(w) for w in words]
    # топ-6 частых суффиксов — цвет, прочие серые
    import collections
    top_sufs = [s for s, _ in collections.Counter(sufs).most_common(6)]
    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=(8, 6))
    for s in set(sufs):
        idx = [i for i, ss in enumerate(sufs) if ss == s]
        col = cmap(top_sufs.index(s) / max(len(top_sufs), 1)) if s in top_sufs else "#cccccc"
        ax.scatter(P[idx, 0], P[idx, 1], s=14, color=col, alpha=0.7,
                   label=s if s in top_sufs else None)
    ax.set_title("Эмбеддинги слов Войнича (UMAP)\nцвет = суффикс (последние 2 глифа)")
    ax.legend(markerscale=2, fontsize=8, loc="best")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(figures / "G_voynich_words.png", dpi=130)
    print(f"  -> {figures/'G_voynich_words.png'}")


if __name__ == "__main__":
    run()
