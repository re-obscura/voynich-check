"""
D: кросс-диалект.

Обучить две отдельные модели — только на фолио Currier A и только на Currier B.
Перплексия:
  модель-A на тест-A  vs  модель-A на тест-B  (и симметрично).
Низкая кросс-перплексия -> та же генеративная машина, только сдвиг словаря;
высокая -> две разные системы. Генеративная переформулировка H3.

Также: перенос внутри B (модель-B на bio vs на остальные секции B).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from voynich_lm.perplexity import cross_entropy_bits
from voynich_lm.train import train_model, load_model


def build_dialect_datasets(ds):
    """
    Разбить фолио Войнича по Currier A/B (по folio_meta). Возвращает
    {dialect: {train_ids, test_ids, test_folios}}.
    test = отложенные фолио того же диалекта (из основного hold-out).
    """
    out = {}
    for dialect in ("A", "B"):
        folios_d = [f for f, m in ds.folio_meta.items()
                    if m["currier"] == dialect and f in ds.train_folios]
        test_d = [f for f, m in ds.folio_meta.items()
                  if m["currier"] == dialect and f in ds.test_folios]
        if not folios_d:
            continue
        train_ids = []
        for f in folios_d:
            train_ids.extend(ds.folio_blocks[f])
            train_ids.append(ds.tok.eos_id)
        test_ids = []
        for f in test_d:
            test_ids.extend(ds.folio_blocks[f])
            test_ids.append(ds.tok.eos_id)
        out[dialect] = {"train_ids": train_ids, "test_ids": test_ids,
                        "train_folios": folios_d, "test_folios": test_d}
    return out


def ensure_dialect_models(ds, ctx, device, verbose=True):
    """Обучить модели 'dialect_A' и 'dialect_B', если их ещё нет."""
    from voynich_lm.model import default_config
    dialects = build_dialect_datasets(ds)
    models = {}
    for dialect, data in dialects.items():
        name = f"dialect_{dialect}"
        try:
            load_model(name, device=device)
            if verbose:
                print(f"  {name}: уже обучена")
        except FileNotFoundError:
            if verbose:
                print(f"  обучение {name} (train={len(data['train_ids'])} глифов)...")
            cfg = default_config(); cfg["dropout"] = 0.2
            train_model(data["train_ids"], ds.tok, name,
                        ctx=ctx, val_ids=data["test_ids"],
                        n_steps=2000, cfg=dict(cfg), device=device, verbose=verbose)
        models[dialect] = data
    return models


def run(ctx: int = 256, device: str | None = None,
        figures: Path = Path("figures"), out: dict | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    figures.mkdir(exist_ok=True)
    from voynich_lm.data import build_voynich_dataset
    ds = build_voynich_dataset()

    print("=== D: кросс-диалект ===")
    dialects = ensure_dialect_models(ds, ctx, device, verbose=False)
    if "A" not in dialects or "B" not in dialects:
        print("  недостаточно фолио обоих диалектов")
        return {}

    # матрица кросс-перплексии: model[i] eval on test[j]
    matrix = {}
    for m_dialect in ("A", "B"):
        model, tok = load_model(f"dialect_{m_dialect}", device=device)
        for t_dialect in ("A", "B"):
            tids = dialects[t_dialect]["test_ids"]
            ce = cross_entropy_bits(model, tids, ctx, device)
            matrix[f"model_{m_dialect}_on_{t_dialect}"] = ce

    # перенос внутри B: bio vs rest
    within_b = {}
    try:
        bio_folios = [f for f, m in ds.folio_meta.items()
                      if m["currier"] == "B" and m["section"] == "bio"]
        rest_folios = [f for f, m in ds.folio_meta.items()
                       if m["currier"] == "B" and m["section"] != "bio"
                       and f in ds.train_folios]
        model_b, _ = load_model("dialect_B", device=device)
        if bio_folios:
            bio_ids = []
            for f in bio_folios:
                bio_ids.extend(ds.folio_blocks[f])
            ce_bio = cross_entropy_bits(model_b, bio_ids, ctx, device)
            within_b["model_B_on_bio"] = ce_bio
        if rest_folios:
            rest_ids = []
            for f in rest_folios:
                rest_ids.extend(ds.folio_blocks[f])
            ce_rest = cross_entropy_bits(model_b, rest_ids, ctx, device)
            within_b["model_B_on_B_nonbio"] = ce_rest
    except Exception as e:
        within_b["error"] = str(e)

    # --- вывод ---
    print("  матрица кросс-энтропии (бит/глиф, ctx={}):".format(ctx))
    print(f"    {'':18s} test-A     test-B")
    print(f"    model_A        {matrix['model_A_on_A']:.3f}      {matrix['model_A_on_B']:.3f}")
    print(f"    model_B        {matrix['model_B_on_A']:.3f}      {matrix['model_B_on_B']:.3f}")
    # симметричный перенос: средняя кросс-перплексия vs собственная
    self_a = matrix["model_A_on_A"]; self_b = matrix["model_B_on_B"]
    cross_ab = matrix["model_A_on_B"]; cross_ba = matrix["model_B_on_A"]
    asym = ((cross_ab - self_b) + (cross_ba - self_a)) / 2
    print(f"  асимметрия переноса (cross − self, среднее): {asym:+.3f} бит/глиф")
    print(f"    (>0 значит: A и B — разные генеративные системы)")
    if within_b:
        if "model_B_on_bio" in within_b and "model_B_on_B_nonbio" in within_b:
            print(f"  внутри B: model_B на bio={within_b['model_B_on_bio']:.3f} "
                  f"vs non-bio={within_b['model_B_on_B_nonbio']:.3f} бит/глиф")

    # --- график: heatmap матрицы ---
    M = np.array([[matrix["model_A_on_A"], matrix["model_A_on_B"]],
                  [matrix["model_B_on_A"], matrix["model_B_on_B"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(M, cmap="YlOrRd")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["test A", "test B"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["model A", "model B"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="black", fontsize=12, fontweight="bold")
    ax.set_title("Кросс-диалект: cross-entropy (бит/глиф)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(figures / "D_cross_dialect.png", dpi=130)
    print(f"  -> {figures/'D_cross_dialect.png'}")

    results = {"matrix": matrix, "asymmetry": asym, "within_B": within_b}
    if out is not None:
        out["cross_dialect"] = results
    return results


if __name__ == "__main__":
    run()
