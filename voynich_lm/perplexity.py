"""
Оценка перплексии / кросс-энтропии на потоке id.
Перплексия = exp(кросс-энтропия в нитах). Для glyph-level это средняя
энтропийная ставка на глиф (бит/глиф = кросс-энтропия по основанию 2).

Ключевая интерпретация: кросс-энтропия LM = оценка истинной энтропийной ставки
с предсказанием по длинному контексту (в отличие от H_{2|1} в entropy.py — порядок 1).
"""
from __future__ import annotations
import math
import numpy as np
import torch

from voynich_lm.data import evaluate_loader


@torch.no_grad()
def cross_entropy_bits(model, ids: list[int], ctx: int, device,
                        batch_size: int = 64) -> float:
    """
    Средняя кросс-энтропия в БИТАХ на глиф по непересекающимся окнам ctx.
    Это оценка энтропийной ставки на длине контекста ctx (бит/глиф).
    """
    if len(ids) < ctx + 1:
        return float("nan")
    X, Y = evaluate_loader(ids, ctx, batch_size, device)
    model.eval()
    total_nll, total_tok = 0.0, 0
    for i in range(0, X.shape[0], batch_size):
        xb, yb = X[i:i+batch_size], Y[i:i+batch_size]
        nll, mask = model.nll_per_position(xb, yb)  # нит, (B,T)
        total_nll += float((nll * mask).sum().item())
        total_tok += float(mask.sum().item())
    # нит -> бит
    return (total_nll / max(total_tok, 1)) / math.log(2)


@torch.no_grad()
def cross_entropy_bits_at_ctx(model, ids, ctx, device, batch_size=64):
    """Алиас, возвращает то же, что cross_entropy_bits (для явности в анализе)."""
    return cross_entropy_bits(model, ids, ctx, device, batch_size)


def perplexity(model, ids, ctx, device, batch_size=64) -> float:
    ce = cross_entropy_bits(model, ids, ctx, device, batch_size)  # бит
    return 2 ** ce if not math.isnan(ce) else float("nan")
