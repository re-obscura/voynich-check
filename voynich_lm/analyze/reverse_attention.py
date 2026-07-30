"""
Направление 4: reverse-LM + анализ attention — атака на H4 «нет синтаксиса».

(a) Обучить LM в ОБРАТНОМ направлении (предсказывать предыдущий глиф по следующим).
    В естественных языках префиксы/суффикты создают направленную асимметрию
    (морфология необратима). Сравнить perplexity forward vs reverse.
    Если Войнич — имитация, асимметрия может быть специфической.

(b) Разобрать attention-матрицы forward-модели: на какие дальние позиции она
    смотрит? Вскрыть нелокальные зависимости, которые NB (H4) не заметил.

(c) «Синтаксис на дальних расстояниях»: дальнобойность attention как функция
    дистанции — отличается ли она от shuffle-null?
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import sys
sys.path.insert(0, ".")
from voynich_lm.data import build_voynich_dataset
from voynich_lm.train import train_model, load_model
from voynich_lm.perplexity import cross_entropy_bits


def ensure_reverse_model(ds, ctx, device, verbose=True):
    """Обучить обратную LM (поток развёрнут)."""
    name = "voynich_reverse"
    try:
        load_model(name, device=device)
        if verbose: print(f"  {name}: уже обучена")
    except FileNotFoundError:
        rev_ids = list(reversed(ds.train_ids))
        rev_val = list(reversed(ds.val_ids))
        from voynich_lm.model import default_config
        cfg = default_config(); cfg["dropout"] = 0.2
        if verbose: print(f"  обучение {name} (reverse stream)...")
        train_model(rev_ids, ds.tok, name, ctx=ctx, val_ids=rev_val,
                    n_steps=2000, cfg=dict(cfg), device=device, verbose=verbose)


@torch.no_grad()
def attention_distance_profile(model, ids: list[int], ctx: int, device,
                                 batch_size: int = 8, n_windows: int = 16) -> np.ndarray:
    """
    Профиль «дальнобойности» attention: усредняем attention-вес (по всем слоям
    и головам) как функцию дистанции query-key. Возвращает массив длины ctx
    (индекс d = средний вес, который позиция отдаёт ключу на дистанции d позади).
    d=0 это самовнимание, d>=1 — предшествующие токены.
    """
    model.eval()
    data = np.array(ids, dtype=np.int64)
    n_win = (len(data) - 1) // ctx
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attn.h
    hd = model.blocks[0].attn.hd
    scale = 1.0 / math.sqrt(hd)
    # аккумулируем «массу по дистанции» отдельно для каждого слоя/головы,
    # потом усредняем. Используем сумму весов и счётчик query-позиций.
    # dist_sum[d] = суммарный attention-вес, отданный ключам на дистанции d,
    # по всем окнам/слоям/головам/query.
    dist_sum = np.zeros(ctx)
    n_query_total = 0  # всего учтённых query-позиций (по всем окнам/слоям/головам)

    for k in range(min(n_win, n_windows)):
        i0 = k * ctx
        x = torch.tensor(data[i0:i0+ctx], device=device, dtype=torch.long)[None]
        T = x.shape[1]
        h = model.tok_emb(x)
        h = model.drop(h)
        for blk in model.blocks:
            ln = blk.ln1(h)
            qkv = blk.attn.qkv(ln).reshape(1, T, 3, n_heads, hd)
            q, kk, _ = qkv.unbind(dim=2)
            q, kk = blk.attn.rope.rotate(q, kk)
            q = q.transpose(1, 2); kk = kk.transpose(1, 2)   # (1,H,T,hd)
            scores = (q @ kk.transpose(-2, -1)) * scale       # (1,H,T,T)
            mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)              # (1,H,T,T)
            # усредним по головам -> (T,T): строки=query, столбцы=key
            attn_mean = attn.mean(1)[0].cpu().numpy()
            # для каждой дистанции d: диагональ query=key+d (d назад)
            for d in range(T):
                diag = np.diagonal(attn_mean, offset=-d) if d > 0 else np.diagonal(attn_mean)
                dist_sum[d] += diag.sum()
            n_query_total += n_heads * (T * T)  # нормировка ниже через сумму dist_sum
            # (нормируем в конце через сумму всех накопленных весов)
    # каждый query отдаёт суммарный вес 1.0; всего query = n_query_positions.
    # но мы накопили по слоям — нормируем так, чтобы сумма по d = 1 (форма профиля).
    total = dist_sum.sum()
    return dist_sum / (total + 1e-9)


import math


def run(ctx: int = 256, device: str | None = None,
        figures: Path = Path("figures"), out: dict | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    figures.mkdir(exist_ok=True)
    print("=" * 68)
    print("НАПРАВЛЕНИЕ 4: reverse-LM + attention (атака на H4)")
    print("=" * 68)

    ds = build_voynich_dataset()
    ensure_reverse_model(ds, ctx, device, verbose=False)

    # --- (a) forward vs reverse perplexity ---
    m_fwd, _ = load_model("voynich", device=device)
    m_rev, _ = load_model("voynich_reverse", device=device)
    ce_fwd = cross_entropy_bits(m_fwd, ds.val_ids, ctx, device)
    # для reverse мерим на развёрнутом val
    ce_rev = cross_entropy_bits(m_rev, list(reversed(ds.val_ids)), ctx, device)

    # референс: то же для shuffle-null (там forward=reverse т.к. нет структуры)
    ce_fwd_sh = cross_entropy_bits(m_fwd, ds.train_ids[:5000], ctx, device)  # грубо

    print(f"\n  (a) Forward vs Reverse perplexity (бит/глиф):")
    print(f"      forward (предсказать следующий): {ce_fwd:.3f}")
    print(f"      reverse (предсказать предыдущий): {ce_rev:.3f}")
    asym = ce_rev - ce_fwd
    print(f"      асимметрия (rev - fwd): {asym:+.3f}")
    if abs(asym) < 0.05:
        print(f"      => асимметрия МАЛА: направленность слабая (как у имитации)")
    else:
        print(f"      => асимметрия ЗНАЧИМА: есть направленная структура (морфология?)")

    # --- (b) attention distance profile ---
    dist = attention_distance_profile(m_fwd, ds.val_ids, ctx=128, device=device)
    # сравнение: у shuffle нет дальней структуры
    m_sh, _ = load_model("shuffle", device=device)
    sh_ids = np.array(ds.train_ids); rng = np.random.default_rng(0); rng.shuffle(sh_ids)
    dist_sh = attention_distance_profile(m_sh, list(sh_ids[:5000]), ctx=128, device=device)

    # нормируем обе кривые
    def norm01(x): return x / (x.sum() + 1e-9)
    dist_n = norm01(dist); dist_sh_n = norm01(dist_sh)

    print(f"\n  (b) Профиль дальнобойности attention (доля веса по дистанции):")
    for d in [1, 2, 4, 8, 16, 32]:
        if d < len(dist_n):
            print(f"      дистанция {d:3d}: Войнич {dist_n[d]:.4f}  shuffle {dist_sh_n[d]:.4f}")
    # «локальность»: доля внимания в радиусе 1-2 vs дальняя
    local_voyn = dist_n[1:4].sum(); local_sh = dist_sh_n[1:4].sum()
    print(f"      локальная масса (d=1..3): Войнич {local_voyn:.3f} vs shuffle {local_sh:.3f}")

    # --- график ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    # forward/reverse
    axes[0].bar(["forward", "reverse"], [ce_fwd, ce_rev],
                color=["#2980b9", "#c0392b"])
    axes[0].set_ylabel("cross-entropy (бит/глиф)")
    axes[0].set_title(f"Направленность: forward vs reverse\nасимметрия {asym:+.3f}")
    axes[0].grid(axis="y", alpha=0.3)
    # attention distance
    dmax = 40
    axes[1].plot(range(1, dmax+1), dist_n[1:dmax+1], "o-", color="#c0392b", label="Voynich")
    axes[1].plot(range(1, dmax+1), dist_sh_n[1:dmax+1], "s-", color="#7f8c8d", label="shuffle-null")
    axes[1].set_xlabel("дистанция (позиций)")
    axes[1].set_ylabel("доля attention-веса")
    axes[1].set_title("Дальнобойность attention (затухание по дистанции)")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "DEC_reverse_attention.png", dpi=130)
    print(f"\n  -> {figures/'DEC_reverse_attention.png'}")

    results = {"forward_ce": ce_fwd, "reverse_ce": ce_rev, "asymmetry": asym,
               "attention_local_mass_voynich": float(local_voyn),
               "attention_local_mass_shuffle": float(local_sh),
               "attention_distance_profile": dist_n.tolist()[:40]}
    if out is not None:
        out["reverse_attention"] = results
    return results


if __name__ == "__main__":
    run()
