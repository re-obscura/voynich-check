"""
Направление 2 (мультимодал) — строгий корпусный тест гипотезы именования.

Идея: если текст описывает иллюстрацию, то слова должны быть СПЕЦИФИЧНЫ для
визуального ТИПА иллюстрации (herbal/bio/astro/...), а не только для диалекта.
Проверяем через декомпозицию:

  1. Внутри одного диалекта: различаются ли секции по словарю сильнее, чем
     случайно (JSD)? Если да — текст привязан к иконографии.
  2. «Маркеры секции»: какие слова сверх-часты в одной секции vs другой
     (внутри диалекта)? Если их немного и они частые -> это словарь тематики.
  3. Контроль: тот же тест на «фиктивных» разбиениях (случайных) — если реальное
     разбиение не сильнее случайного, привязки нет.

Это строгий, воспроизводимый тест именования на полном корпусе, не требующий
ручного маппинг изображение↔текст. Дополнительно (при наличии факсимиле)
VLM-описания растений можно использовать для качественного подтверждения.
"""
from __future__ import annotations
import json
import collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, ".")
from voi.parse_ivtff import parse_ivtff, add_n_token_columns
from voi.common import data_path, js_divergence, freq_dist


def section_vocab_within_dialect(lines_df, dialect: str, sections: list[str]) -> dict:
    """Словари секций внутри одного диалекта -> {section: freq_dist}."""
    out = {}
    for sec in sections:
        sub = lines_df[(lines_df["section"] == sec) & (lines_df["currier"] == dialect) &
                       (lines_df["n_tokens"] > 0)]
        toks = [t for ts in sub["tokens"] for t in ts]
        out[sec] = freq_dist(toks) if toks else {}
    return out


def vocab_jsd_matrix(vocabs: dict) -> np.ndarray:
    """Матрица JSD между словарями секций."""
    secs = list(vocabs.keys())
    n = len(secs)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = js_divergence(vocabs[secs[i]], vocabs[secs[j]])
    return M, secs


def random_split_jsd(lines_df, dialect: str, section: str, n_iter: int = 1000,
                      seed: int = 7) -> tuple[float, float]:
    """Null: разбиваем одну секцию случайным образом на 2 половины, мерим JSD.
    Возвращает (mean, std) распределения JSD при случайном разбиении."""
    sub = lines_df[(lines_df["section"] == section) & (lines_df["currier"] == dialect) &
                   (lines_df["n_tokens"] > 0)]
    toks_by_line = [list(ts) for ts in sub["tokens"]]
    if len(toks_by_line) < 10:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    jsds = []
    for _ in range(n_iter):
        perm = list(range(len(toks_by_line)))
        rng.shuffle(perm)
        half = len(perm) // 2
        a = [t for i in perm[:half] for t in toks_by_line[i]]
        b = [t for i in perm[half:] for t in toks_by_line[i]]
        jsds.append(js_divergence(freq_dist(a), freq_dist(b)))
    return float(np.mean(jsds)), float(np.std(jsds))


def section_markers(lines_df, dialect: str, sec_a: str, sec_b: str, topk: int = 10) -> dict:
    """Слова, сверх-частые в sec_a vs sec_b (внутри диалекта).
    Маркер = log(P(w|A)/P(w|B)), сглаженный."""
    va = section_vocab_within_dialect(lines_df, dialect, [sec_a])[sec_a]
    vb = section_vocab_within_dialect(lines_df, dialect, [sec_b])[sec_b]
    keys = set(va) | set(vb)
    markers = []
    eps = 1e-4
    for w in keys:
        pa = va.get(w, eps); pb = vb.get(w, eps)
        # только достаточно частые
        if va.get(w, 0) > 0.003 or vb.get(w, 0) > 0.003:
            markers.append((w, np.log(pa / pb), pa, pb))
    markers.sort(key=lambda x: -x[1])
    top_a = [(w, r, pa) for w, r, pa, pb in markers[:topk]]
    markers.sort(key=lambda x: x[1])
    top_b = [(w, r, pa) for w, r, pa, pb in markers[:topk]]
    return {"sec_a": sec_a, "sec_b": sec_b, "top_in_A": top_a, "top_in_B": top_b}


def run(figures: Path = Path("figures"), out: dict | None = None) -> dict:
    figures.mkdir(exist_ok=True)
    print("=" * 68)
    print("НАПРАВЛЕНИЕ 2: строгий тест гипотезы именования (корпусный)")
    print("=" * 68)
    print()
    print("Вопрос: слова специфичны для ТИПА иллюстрации (herbal/bio/...) ВНУТРИ")
    print("одного диалекта? Если да — текст привязан к иконографии (имя-описание).")
    print()

    ld, _ = parse_ivtff(str(data_path("voynich")))
    ld = add_n_token_columns(ld)

    results = {}
    # Currier B: 3 секции (herbal, bio, recipes) — достаточно мощно
    print("--- Currier B: словари секций herbal / bio / recipes ---")
    secB = section_vocab_within_dialect(ld, "B", ["herbal", "bio", "recipes"])
    for s, v in secB.items():
        print(f"  {s:10s}: {sum(int(c*1000000) for c in v.values())//1000}K токенов (прибл.)")
    M, secs = vocab_jsd_matrix(secB)
    print(f"\n  JSD между словарями секций (внутри B):")
    print(f"  {'':10s} " + " ".join(f"{s:>10s}" for s in secs))
    for i, s in enumerate(secs):
        print(f"  {s:10s} " + " ".join(f"{M[i,j]:10.3f}" for j in range(len(secs))))

    # null: случайное разбиение herbal на 2 половины
    null_mean, null_std = random_split_jsd(ld, "B", "herbal", n_iter=1000)
    print(f"\n  Null (случайное разбиение herbal на 2 половины): JSD = {null_mean:.3f} ± {null_std:.3f}")
    for i in range(len(secs)):
        for j in range(i+1, len(secs)):
            real = M[i, j]
            z = (real - null_mean) / null_std if null_std > 0 else float("nan")
            verdict = "ЗНАЧИМО выше null" if z > 3 else "в диапазоне случайного"
            print(f"  JSD({secs[i]},{secs[j]}) = {real:.3f}  z={z:+.1f}  -> {verdict}")

    # маркеры секций
    print(f"\n  Маркеры секций (сверх-частые слова, внутри B):")
    for pair in [("bio", "herbal"), ("recipes", "herbal")]:
        mrk = section_markers(ld, "B", pair[0], pair[1], topk=8)
        print(f"\n    топ-слов в {pair[0]} (vs {pair[1]}):")
        for w, r, p in mrk["top_in_A"]:
            print(f"      {w:12s} log-ratio={r:+.2f}  P={p:.4f}")
        print(f"    топ-слов в {pair[1]} (vs {pair[0]}):")
        for w, r, p in mrk["top_in_B"]:
            print(f"      {w:12s} log-ratio={r:+.2f}  P={p:.4f}")

    results = {"jsd_matrix_B": {secs[i]: {secs[j]: float(M[i,j]) for j in range(len(secs))}
                                for i in range(len(secs))},
               "null_random_jsd": {"mean": null_mean, "std": null_std}}

    # --- график: heatmap JSD ---
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(M, cmap="YlOrRd")
    ax.set_xticks(range(len(secs))); ax.set_xticklabels(secs, rotation=20)
    ax.set_yticks(range(len(secs))); ax.set_yticklabels(secs)
    for i in range(len(secs)):
        for j in range(len(secs)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if M[i,j] > 0.5 else "black")
    ax.set_title("JSD словарей секций (внутри Currier B)\n+ null = случайное разбиение")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(figures / "DEC_naming.png", dpi=130)
    print(f"\n  -> {figures/'DEC_naming.png'}")

    if out is not None:
        out["naming_test"] = results
    return results


if __name__ == "__main__":
    run()
