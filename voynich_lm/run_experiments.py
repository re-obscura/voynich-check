"""
Оркестратор: обучает семейство моделей (если чекпоинтов нет) и прогоняет все
анализы A–G. Сохраняет результаты в results_lm.json и figures/.

Использование:
    python -m voynich_lm.run_experiments           # всё
    python -m voynich_lm.run_experiments --train-only
    python -m voynich_lm.run_experiments --no-train --analyses A D   # только анализ
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from voynich_lm.data import build_voynich_dataset, build_control_dataset
from voynich_lm.train import train_model, load_model


def ensure_models(ctx=256, n_steps=2000, device=None, verbose=True):
    """Обучить всё семейство, если чекпоинтов нет. Early-stopping по val_loss."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    from voi.common import data_path
    from voynich_lm.model import default_config
    ds = build_voynich_dataset()
    target = len(ds.train_ids)
    # конфиг с повышенной регуляризацией (маленький корпус -> иначе запоминает)
    cfg = default_config()
    cfg["dropout"] = 0.2

    def have(name):
        try:
            load_model(name, device=device); return True
        except FileNotFoundError:
            return False

    # 1) voynich
    if not have("voynich"):
        print("\n# Обучение модели: voynich")
        train_model(ds.train_ids, ds.tok, "voynich", ctx=ctx, val_ids=ds.val_ids,
                    n_steps=n_steps, device=device, verbose=verbose, cfg=dict(cfg))
    # 2) controls
    specs = [("kjv", str(data_path("kjv"))),
             ("moby", str(data_path("moby"))),
             ("paradise", str(data_path("paradise"))),
             ("caesar", str(data_path("caesar")))]
    for cname, cpath in specs:
        if have(cname):
            continue
        print(f"\n# Обучение модели: {cname}")
        tok, ids = build_control_dataset(cname, target_chars=target, text_path=cpath)
        ntr = int(len(ids) * 0.85)
        train_model(ids[:ntr], tok, cname, ctx=ctx, val_ids=ids[ntr:],
                    n_steps=n_steps, device=device, verbose=verbose, cfg=dict(cfg))
    # 3) shuffle-null
    if not have("shuffle"):
        print("\n# Обучение модели: shuffle (null)")
        tok, ids = build_control_dataset("shuffle", target_chars=target)
        ntr = int(len(ids) * 0.85)
        train_model(ids[:ntr], tok, "shuffle", ctx=ctx, val_ids=ids[ntr:],
                    n_steps=n_steps, device=device, verbose=verbose, cfg=dict(cfg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--n-steps", type=int, default=2000)
    ap.add_argument("--no-train", action="store_true",
                    help="пропустить обучение (использовать готовые чекпоинты)")
    ap.add_argument("--train-only", action="store_true",
                    help="только обучение, без анализов")
    ap.add_argument("--analyses", nargs="*", default=None,
                    help="подмножество анализов: A B D E F G (по умолчанию все)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    figures = Path("figures")
    results = {}

    if not args.no_train:
        ensure_models(ctx=args.ctx, n_steps=args.n_steps, device=device)
    if args.train_only:
        return

    analyses = args.analyses or ["A", "B", "D", "E", "F", "G"]
    from voynich_lm.analyze import (entropy_generalization as A,
                                      generative as B,
                                      cross_dialect as D,
                                      position as E,
                                      probing as F,
                                      embeddings as G)
    table = {"A": ("энтропия + обобщение", A.run),
             "B": ("генеративный тест", B.run),
             "D": ("кросс-диалект", D.run),
             "E": ("позиция", E.run),
             "F": ("пробы", F.run),
             "G": ("эмбеддинги", G.run)}

    # какие аргументы принимает каждый анализ. ctx — только для тех, у кого он есть.
    ctx_analyses = {"A", "D", "E"}

    for key in analyses:
        if key not in table:
            print(f"неизвестный анализ: {key}"); continue
        label, fn = table[key]
        print(f"\n>>> {key}: {label}")
        try:
            if key in ctx_analyses:
                fn(ctx=args.ctx, device=device, figures=figures, out=results)
            else:
                fn(device=device, figures=figures, out=results)
        except Exception as e:
            import traceback
            print(f"  [провал] {key}: {e}")
            traceback.print_exc()
            results[key] = {"error": str(e)}

    # сериализация
    def default(o):
        try:
            return float(o)
        except (TypeError, ValueError):
            try:
                return list(o)
            except TypeError:
                return str(o)
    Path("results_lm.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=default),
        encoding="utf-8")
    print("\n# Результаты сохранены в results_lm.json")


if __name__ == "__main__":
    main()
