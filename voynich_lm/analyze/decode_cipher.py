"""
Направление 1: дешифровка как задача оптимизации (LM = оракул).

Перебираем отображения EVA-глиф -> целевой символ (a-z, +пробел/sep). Для каждой
кандидатной перестановки расшифровываем подмножество корпуса Войнича и меряем
perplexity моделью целевого языка (KJV/Latin). Если есть ключ, при котором
«расшифрованный» Войнич получает АНОМАЛЬНО низкую perplexity (намного ниже
random-перестановки) — это кандидат.

Ключевая логика: simulated annealing по пространству перестановок.
- Состояние: биекция subset(EVA-глифов) -> целевые символы.
- Шаг: swap двух целевых символов.
- Фитнес: средняя кросс-энтропия модели целевого языка на расшифровке.
- Температура падает по расписанию.

Если «имитация без смысла» верна -> гладкого минимума не будет, perplexity
останется в диапазоне случайных перестановок. Это и есть отрицательный результат:
строгая проверка того, что никакой простой подстановочный шифр не работает.
"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, ".")
from voynich_lm.train import load_model
from voi.parse_ivtff import parse_ivtff, add_n_token_columns
from voi.common import data_path


def get_voynich_char_stream(max_chars: int = 5000) -> str:
    """Поток глифов Войнича (только реальные глифы + '.' разделитель)."""
    ld, _ = parse_ivtff(str(data_path("voynich")))
    ld = add_n_token_columns(ld)
    txt = ld[ld["n_tokens"] > 0]
    chars = []
    for toks in txt["tokens"]:
        for w in toks:
            for c in w:
                chars.append(c)
            chars.append(".")
    return "".join(chars)[:max_chars]


@torch.no_grad()
def ce_of_decoded(model, tok, decoded_str: str, ctx: int, device) -> float:
    """Кросс-энтропия (нит) модели на расшифрованной строке. None если невалидно."""
    ids = tok.encode(decoded_str)
    if len(ids) < ctx + 1:
        return float("nan")
    model.eval()
    # оценим по непересекающимся окнам
    n_win = (len(ids) - 1) // ctx
    total_nll, total_tok = 0.0, 0
    for k in range(n_win):
        i0 = k * ctx
        x = torch.tensor(ids[i0:i0+ctx], device=device, dtype=torch.long)[None]
        tgt = torch.tensor(ids[i0+1:i0+1+ctx], device=device, dtype=torch.long)[None]
        nll, mask = model.nll_per_position(x, tgt)
        total_nll += float((nll * mask).sum())
        total_tok += float(mask.sum())
    return total_nll / max(total_tok, 1)


def apply_perm(stream: str, perm: dict) -> str:
    """Перевести поток глифов через биекцию perm. Неизвестные -> '.'"""
    return "".join(perm.get(c, ".") for c in stream)


def simulated_annealing(stream: str, target_model_name: str,
                         evo_glyphs: list[str], target_chars: list[str],
                         ctx: int = 96, device: str = "cpu",
                         n_iter: int = 4000, seed: int = 7) -> dict:
    """
    SA по перестановкам evo_glyphs -> target_chars.
    Возвращает {best_perm, best_ce, history, baseline_random_ce}.
    """
    rng = np.random.default_rng(seed)
    model, tok = load_model(target_model_name, device=device)

    # стартовая случайная биекция
    targets = list(target_chars)
    rng.shuffle(targets)
    perm = dict(zip(evo_glyphs, targets[:len(evo_glyphs)]))
    decoded = apply_perm(stream, perm)
    cur_ce = ce_of_decoded(model, tok, decoded, ctx, device)
    best_perm, best_ce = dict(perm), cur_ce

    history = [cur_ce]
    T0, Tend = 0.3, 0.005
    for it in range(n_iter):
        T = T0 * (Tend / T0) ** (it / n_iter)
        # шаг: swap двух целевых символов в перестановке
        new_perm = dict(perm)
        a, b = rng.choice(evo_glyphs, 2, replace=False)
        new_perm[a], new_perm[b] = new_perm[b], new_perm[a]
        new_decoded = apply_perm(stream, new_perm)
        new_ce = ce_of_decoded(model, tok, new_decoded, ctx, device)
        if math.isnan(new_ce):
            continue
        # accept?
        dE = new_ce - cur_ce
        if dE < 0 or rng.random() < math.exp(-dE / T):
            perm, cur_ce = new_perm, new_ce
            if cur_ce < best_ce:
                best_perm, best_ce = dict(perm), cur_ce
        history.append(cur_ce)

    # baseline: средняя CE при N случайных перестановках
    random_ces = []
    for _ in range(30):
        t = list(target_chars); rng.shuffle(t)
        rp = dict(zip(evo_glyphs, t[:len(evo_glyphs)]))
        random_ces.append(ce_of_decoded(model, tok, apply_perm(stream, rp), ctx, device))
    baseline_random = float(np.nanmean(random_ces))

    return {"best_perm": best_perm, "best_ce": best_ce,
            "baseline_random_ce": baseline_random,
            "history": history, "n_iter": n_iter}


def run(figures: Path = Path("figures"), out: dict | None = None) -> dict:
    figures.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 68)
    print("НАПРАВЛЕНИЕ 1: LM-подбор шифра (simulated annealing)")
    print("=" * 68)

    stream = get_voynich_char_stream(max_chars=4000)
    # EVA-глифы (частые, без редких v/x/z/g)
    evo_glyphs = ["a", "c", "d", "e", "h", "i", "k", "l", "o", "r",
                  "s", "t", "y", "n", "m", "p", "q"]
    target_chars = [c for c in "abcdefghijklmnopqrstuvwxyz "]

    results = {}
    for tgt in ["kjv"]:
        print(f"\n  Целевой язык: {tgt} (перебор {len(evo_glyphs)}! перестановок, SA)")
        print(f"  окно ctx=96, 4000 итераций SA...")
        t0 = time.time()
        r = simulated_annealing(stream, tgt, evo_glyphs, target_chars,
                                 ctx=96, device=device, n_iter=4000)
        dt = time.time() - t0
        improvement = r["baseline_random_ce"] - r["best_ce"]
        print(f"  выполнено за {dt:.0f}s")
        print(f"  best CE (нит): {r['best_ce']:.4f}")
        print(f"  baseline random CE: {r['baseline_random_ce']:.4f}")
        print(f"  улучшение: {improvement:+.4f} нит ({improvement/r['baseline_random_ce']*100:+.1f}%)")
        # оценка значимости: SA-улучшение аномально?
        z = improvement / (0.02 + 0.0) if improvement > 0 else 0  # грубо
        print(f"  найденный ключ (топ-EVA -> латиница):")
        for g in sorted(r["best_perm"]):
            print(f"    {g} -> '{r['best_perm'][g]}'")
        verdict = ("АНОМАЛЬНОЕ улучшение — кандидат на ключ" if improvement > 0.3
                   else "улучшение в диапазоне случайного — простой шифр НЕ работает")
        print(f"\n  ВЕРДИКТ: {verdict}")
        results[tgt] = {"best_ce": r["best_ce"], "baseline_random_ce": r["baseline_random_ce"],
                        "improvement": improvement, "best_perm": r["best_perm"]}

        # --- график: кривая SA + baseline ---
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(r["history"], color="#2980b9", lw=0.8, alpha=0.7, label="SA current CE")
        ax.axhline(r["baseline_random_ce"], color="#7f8c8d", ls="--",
                   label=f"random baseline = {r['baseline_random_ce']:.3f}")
        ax.axhline(r["best_ce"], color="#c0392b", ls="--",
                   label=f"SA best = {r['best_ce']:.3f}")
        ax.set_xlabel("итерация SA")
        ax.set_ylabel("cross-entropy (нит) на расшифровке")
        ax.set_title(f"LM-подбор шифра Войнича → {tgt}\n"
                     f"улучшение {improvement:+.3f} нит ({improvement/r['baseline_random_ce']*100:+.1f}%)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures / "DEC_cipher_search.png", dpi=130)
        print(f"  -> {figures/'DEC_cipher_search.png'}")

    if out is not None:
        out["cipher_search"] = results
    return results


if __name__ == "__main__":
    run()
