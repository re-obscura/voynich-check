"""
A+C: истинная энтропийная ставка на длинном контексте + зазор обобщения.

A) Кросс-энтропия LM на длинном контексте (ctx=256) — расширяет voi/entropy.py
   (там только H_{2|1}, порядок 1) до оценки энтропии на больших расстояниях.
   Сравнение: Войнич vs 4 природных языка vs shuffle-null, все той же длины.
C) Зазор обобщения: perplexity(train) vs perplexity(test-фолио).
   Где Войнич на шкале «язык обобщает ↔ шум зубрит»?
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from voynich_lm.perplexity import cross_entropy_bits
from voynich_lm.train import load_model


def evaluate_language(name: str, train_ids, test_ids, ctx, device) -> dict:
    """Для модели name: train/test cross-entropy (бит/глиф) при данном ctx."""
    model, tok = load_model(name, device=device)
    ce_train = cross_entropy_bits(model, train_ids, ctx, device)
    ce_test = cross_entropy_bits(model, test_ids, ctx, device)
    return {"name": name, "ctx": ctx,
            "ce_train_bits": ce_train, "ce_test_bits": ce_test,
            "ppl_train": 2 ** ce_train if ce_train == ce_train else float("nan"),
            "ppl_test": 2 ** ce_test if ce_test == ce_test else float("nan"),
            "gen_gap": ce_test - ce_train}


def run(ctx: int = 256, device: str | None = None, figures: Path = Path("figures"),
        out: dict | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    figures.mkdir(exist_ok=True)

    # соберём датасеты для всех языков (control-фолд вычисляется на месте)
    from voynich_lm.data import build_voynich_dataset, build_control_dataset
    ds = build_voynich_dataset()
    target = len(ds.train_ids)
    # test_ids для Войнича: его val (отложенные фолио)
    voyn_test = ds.val_ids

    # build_control_dataset возвращает (tok, ids). Разобьём на train/test 85/15.
    def split_train_test(ids, frac=0.85, seed=7):
        rng = np.random.default_rng(seed)
        ids = list(ids)
        n = int(len(ids) * frac)
        # непрерывный разрез (как фолио-блоки) — для честности
        return ids[:n], ids[n:]

    languages = {}
    # voynich
    languages["voynich"] = (ds.train_ids, voyn_test, ds.tok)
    # controls
    from voi.common import data_path
    control_specs = [
        ("kjv", str(data_path("kjv"))),
        ("moby", str(data_path("moby"))),
        ("paradise", str(data_path("paradise"))),
        ("caesar", str(data_path("caesar"))),
    ]
    control_data = {}
    for cname, cpath in control_specs:
        tok, ids = build_control_dataset(cname, target_chars=target, text_path=cpath)
        tr, te = split_train_test(ids)
        control_data[cname] = (tr, te, tok)
    # shuffle-null: тот же токенайзер и поток, что у Войнича, перемешанный
    tok_s, ids_s = build_control_dataset("shuffle", target_chars=target)
    control_data["shuffle"] = (ids_s[:int(len(ids_s)*0.85)], ids_s[int(len(ids_s)*0.85):], tok_s)

    results = {}
    print("=== A+C: энтропийная ставка (бит/глиф) + зазор обобщения ===")
    # voynich
    r = evaluate_language("voynich", ds.train_ids, voyn_test, ctx, device)
    results["voynich"] = r
    print(f"  voynich     ctx={ctx}: train={r['ce_train_bits']:.3f} test={r['ce_test_bits']:.3f} "
          f"bit/glyph  (gap={r['gen_gap']:+.3f})")
    # controls (только если модели обучены — иначе пропустим)
    for cname in ["kjv", "moby", "paradise", "caesar", "shuffle"]:
        try:
            tr, te, _ = control_data[cname]
            r = evaluate_language(cname, tr, te, ctx, device)
            results[cname] = r
            print(f"  {cname:10s} ctx={ctx}: train={r['ce_train_bits']:.3f} test={r['ce_test_bits']:.3f} "
                  f"bit/glyph  (gap={r['gen_gap']:+.3f})")
        except FileNotFoundError:
            print(f"  {cname:10s}: модель не обучена, пропуск")

    # --- график: test cross-entropy + gap по языкам ---
    names = list(results.keys())
    ce_test = [results[n]["ce_test_bits"] for n in names]
    gaps = [results[n]["gen_gap"] for n in names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#c0392b" if n == "voynich" else ("#7f8c8d" if n == "shuffle" else "#2980b9")
              for n in names]
    ax1.bar(names, ce_test, color=colors)
    ax1.set_ylabel("test cross-entropy (бит/глиф)")
    ax1.set_title(f"Энтропийная ставка (ctx={ctx})\n← ниже = предсказуемее")
    ax1.tick_params(axis="x", rotation=30)
    ax2.bar(names, gaps, color=colors)
    ax2.set_ylabel("gen gap = test − train (бит/глиф)")
    ax2.set_title("Зазор обобщения\n← выше = больше запоминания")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(figures / "A_entropy_generalization.png", dpi=130)
    print(f"  -> {figures/'A_entropy_generalization.png'}")

    if out is not None:
        out["entropy_generalization"] = results
    return results


if __name__ == "__main__":
    res = run()
    Path("results_lm.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
