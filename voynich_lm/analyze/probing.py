"""
F: линейные пробы скрытых состояний + инспекция attention.

Скрытое состояние фолио = средний pooled-вектор по позициям (из последнего
блока). Обучаем логистическую регрессию предсказывать section / currier / scribe.
Ожидание (согласованное с H3): currier извлекается хорошо, section/scribe — нет.

Attention: усреднённые по головам/слоям матрицы внимания — на предмет
line-edge (диагональных) / диалектных паттернов. Не углубляемся, только
визуализируем общий отпечаток.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from voynich_lm.train import load_model


@torch.no_grad()
def folio_hidden_vectors(ds, model, device, ctx: int = 256) -> tuple[np.ndarray, list[str]]:
    """
    Pooled hidden state для каждого фолио: усредняем скрытые по непересекающимся
    окнам и позициям. Возвращает (X [n_folios, d], folios).
    """
    model.eval()
    X = []
    folios = sorted(ds.folio_blocks.keys())
    for f in folios:
        ids = ds.folio_blocks[f]
        if len(ids) < ctx:
            ids = ids + [ds.tok.eos_id] * (ctx - len(ids) + 1)
        n_win = (len(ids) - 1) // ctx
        if n_win == 0:
            n_win = 1
        vecs = []
        for k in range(n_win):
            i0 = k * ctx
            x = torch.tensor(ids[i0:i0+ctx], device=device, dtype=torch.long)[None]
            _, _, h = model(x, return_hidden=True)   # (1,ctx,d)
            vecs.append(h[0].mean(0).cpu().numpy())
        X.append(np.mean(vecs, axis=0))
    return np.stack(X), folios


def probe_labels(ds, folios):
    """Возвращает три массива меток: section, currier, scribe."""
    def col(key):
        return np.array([ds.folio_meta.get(f, {}).get(key, "?") for f in folios])
    return col("section"), col("currier"), col("scribe")


def run_probe(X, labels, name, min_class_size: int = 3):
    """5-fold stratified CV логрег. Возвращает accuracy и baseline."""
    # оставим классы с >= min_class_size
    vals, counts = np.unique(labels, return_counts=True)
    keep = {v for v, c in zip(vals, counts) if c >= min_class_size and v != "?"}
    mask = np.array([l in keep for l in labels])
    Xs, ys = X[mask], labels[mask]
    if len(set(ys)) < 2 or len(ys) < 6:
        return {"name": name, "acc": float("nan"), "baseline": float("nan"),
                "n": int(len(ys)), "n_classes": int(len(set(ys)))}
    le = {v: i for i, v in enumerate(sorted(set(ys)))}
    y = np.array([le[v] for v in ys])
    baseline = max(np.bincount(y)) / len(y)
    n_splits = min(5, len(set(ys)), np.min(np.bincount(y)))
    if n_splits < 2:
        return {"name": name, "acc": float("nan"), "baseline": float(baseline),
                "n": int(len(ys)), "n_classes": int(len(set(ys)))}
    try:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        scores = cross_val_score(clf, Xs, y, cv=kf)
        return {"name": name, "acc": float(np.mean(scores)),
                "baseline": float(baseline), "n": int(len(ys)),
                "n_classes": int(len(set(ys)))}
    except Exception as e:
        return {"name": name, "acc": float("nan"), "baseline": float(baseline),
                "n": int(len(ys)), "error": str(e)}


def run(device: str | None = None,
        figures: Path = Path("figures"), out: dict | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    figures.mkdir(exist_ok=True)
    from voynich_lm.data import build_voynich_dataset
    ds = build_voynich_dataset()
    model, _ = load_model("voynich", device=device)

    print("=== F: пробы скрытых состояний ===")
    X, folios = folio_hidden_vectors(ds, model, device)
    sec, cur, scr = probe_labels(ds, folios)
    results = {}
    for name, labels in [("section", sec), ("currier", cur), ("scribe", scr)]:
        r = run_probe(X, labels, name)
        results[name] = r
        acc = r["acc"]
        bl = r["baseline"]
        acc_s = f"{acc*100:.1f}%" if acc == acc else "—"
        bl_s = f"{bl*100:.1f}%"
        print(f"  probe[{name:8s}]: acc={acc_s}  baseline={bl_s}  "
              f"(n={r['n']}, классов={r['n_classes']})")
    print("  (ожидание из H3: currier извлекается хорошо, section/scribe — нет)")

    # --- график ---
    names = list(results.keys())
    accs = [results[n]["acc"] if results[n]["acc"] == results[n]["acc"] else 0 for n in names]
    baselines = [results[n]["baseline"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - 0.2, accs, 0.4, label="probe accuracy", color="#2980b9")
    ax.bar(x + 0.2, baselines, 0.4, label="baseline (most-frequent)", color="#bdc3c7")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("accuracy (5-fold CV)")
    ax.set_title("Линейные пробы скрытых состояний фолио\n(согласованность с H3)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "F_probing.png", dpi=130)
    print(f"  -> {figures/'F_probing.png'}")

    if out is not None:
        out["probing"] = results
    return results


if __name__ == "__main__":
    run()
