"""
Раунд 5: текст как физический объект — три новых гипотезы.

G1. WORD-WRAP RECONSTRUCTION (новая).
    Рукопись — страницы со строками фиксированной ширины. Если текст ПИСАЛСЯ
    (под страницу), разбиение слов по строкам подчиняется модели жадного
    word-wrap: каждая строка заполняется словами, пока сумма их «длин» (в глифах
    или в визуальной ширине) не достигает порога. Реконструируем этот порог и
    проверяем, насколько точно он предсказывает реальные границы строк.

G2. EFFECTIVE MEMORY LENGTH (строгий PACF).
    Несколько раундов говорили про «локальность»/«дальнобойность» нестрого.
    Здесь — partial autocorrelation function потока глифов: первый лаг, где
    PACF обрывается = точный порядок памяти процесса. Это разделяет «короткая
    слоговая память» и «длинная синтаксическая».

G3. GENERATIVE SUFFICIENCY (строгая модель имитации).
    Выучить ПРОСТЕЙШУЮ слоговую модель (n-грамма глифов низкого порядка) и
    проверить, воспроизводит ли она ВСЮ статистику Войнича (энтропия, Ципф,
    line-edge, MI). Если да — это строгая модель «имитации»: вся сложность
    объясняется короткой слоговой памятью, никакого синтаксиса не нужно.
"""
from __future__ import annotations
import json
import math
import collections
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, ".")
from voi.parse_ivtff import parse_ivtff, add_n_token_columns
from voi.common import data_path, js_divergence, freq_dist


def load_voynich():
    ld, _ = parse_ivtff(str(data_path("voynich")))
    ld = add_n_token_columns(ld)
    return ld


# ============================================================
# G1. WORD-WRAP RECONSTRUCTION
# ============================================================

def g1_word_wrap(lines_df) -> dict:
    """
    Модель: строка = последовательность слов, разделённых точкой.
    «Длина слова» = число глифов (прокси визуальной ширины).
    Строка заполняется, пока суммарная длина не достигает порога W.

    Главный тест — ТОЧНОСТЬ ВОССТАНОВЛЕНИЯ ГРАНИЦ (а не числа строк):
    жадным word-wrap'ом предсказываем, какие слова окажутся ПОСЛЕДНИМИ в строке
    (позиции переносов), и сравниваем с реальными. Shuffle слов должен
    предсказывать границы гораздо хуже — если только число строк совпадает
    случайно, то позиции переносов у shuffle будут рассинхронизированы.
    """
    P = lines_df[lines_df["locus_type"].str.startswith("P") &
                 (lines_df["n_tokens"] > 0)].copy()
    # длина строки = сумма длин слов + точки-разделители
    line_lengths = []
    line_wordlens = []
    for toks in P["tokens"]:
        wl = [len(t) for t in toks]
        total = sum(wl) + (len(wl) - 1)
        line_lengths.append(total)
        line_wordlens.append(wl)
    line_lengths = np.array(line_lengths, dtype=float)
    W_est = float(np.median(line_lengths))

    # корреляция длина строки / длина последнего слова
    last_w = np.array([wl[-1] if wl else 0 for wl in line_wordlens], dtype=float)
    corr = float(np.corrcoef(line_lengths, last_w)[0, 1])

    # ---- Главный тест: восстановление ПОЗИЦИЙ переносов ----
    # По каждой странице: поток слов с пометками реальных концов строк.
    # word-wrap предсказывает концы строк; точность = доля совпавших границ.
    page_data = collections.defaultdict(list)  # folio -> [(word, is_line_end), ...]
    for f, toks, ltype in zip(P["folio"], P["tokens"], P["locus_type"]):
        if ltype.startswith("P") and toks:
            for i, w in enumerate(toks):
                page_data[f].append((w, i == len(toks) - 1))

    def wrap_boundaries(words, W):
        """Возвращает множество индексов слов, НА которых строка заканчивается."""
        ends = set(); cur = 0; last = 0
        for i, w in enumerate(words):
            wl = len(w)
            add = wl if cur == 0 else wl + 1
            cur += add
            # заканчиваем строку, если следующее слово не помещается
            nxt = words[i+1] if i+1 < len(words) else None
            nxt_add = (len(nxt) + 1) if nxt else 999
            if cur >= W * 0.85 or (nxt and cur + nxt_add > W):
                ends.add(i); cur = 0
        return ends

    def boundary_f1(real_ends, pred_ends, n):
        if not real_ends and not pred_ends: return 1.0
        tp = len(real_ends & pred_ends)
        prec = tp / len(pred_ends) if pred_ends else 0
        rec = tp / len(real_ends) if real_ends else 0
        return 2*prec*rec/(prec+rec) if (prec+rec) else 0

    rng = np.random.default_rng(7)
    W_int = int(round(W_est))
    f1_real, f1_null = [], []
    for f, pairs in list(page_data.items()):
        if len(pairs) < 10: continue
        words = [w for w, _ in pairs]
        real_ends = {i for i, (_, e) in enumerate(pairs) if e}
        pred = wrap_boundaries(words, W_int)
        f1_real.append(boundary_f1(real_ends, pred, len(words)))
        # null: shuffle
        ws = list(words); rng.shuffle(ws)
        pred_n = wrap_boundaries(ws, W_int)
        f1_null.append(boundary_f1(real_ends, pred_n, len(words)))
    f1_real = float(np.mean(f1_real)) if f1_real else 0
    f1_null = float(np.mean(f1_null)) if f1_null else 0

    return {"W_estimated_glyphs": W_est, "line_length_median": W_est,
            "line_length_mean": float(np.mean(line_lengths)),
            "line_length_std": float(np.std(line_lengths)),
            "line_length_cv": float(np.std(line_lengths) / np.mean(line_lengths)),
            "corr_len_lastword": corr,
            "boundary_f1_real": f1_real, "boundary_f1_shuffle": f1_null,
            "line_length_hist": np.histogram(line_lengths, bins=30)[0].tolist()}


# ============================================================
# G2. EFFECTIVE MEMORY LENGTH (PACF)
# ============================================================

def g2_pacf_memory(lines_df, max_lag: int = 32) -> dict:
    """
    Partial autocorrelation function потока глифов.
    PACF обрывается (падает в шум) после лага k => порядок памяти процесса = k.
    В естественном языке PACF затухает медленно (длинная память = синтаксис);
    в локальном слоговом генераторе — резко обрывается после малого k.
    """
    P = lines_df[lines_df["n_tokens"] > 0].copy()
    # кодируем глифы как целые; соберём поток
    all_glyphs = []
    for toks in P["tokens"]:
        for w in toks:
            all_glyphs.extend(list(w))
            all_glyphs.append(".")  # разделитель
    # простой код: глиф -> 0..V-1
    chars = sorted(set(all_glyphs))
    cmap = {c: i for i, c in enumerate(chars)}
    stream = np.array([cmap[c] for c in all_glyphs], dtype=float)
    # PACF по лагам (классический алгоритм Юла-Уокера через регрессии)
    pacf = np.zeros(max_lag)
    n = len(stream)
    mean = stream.mean()
    for k in range(1, max_lag + 1):
        # PACF(k) = коэффициент при x_{t-k} в регрессии x_t на x_{t-1}..x_{t-k}
        if n - k < k + 10: break
        X = np.column_stack([stream[k:n] - mean] + [stream[k-i-1:n-i-1] - mean for i in range(k)])
        # последняя колонка = x_{t-k}; её стандартизированный коэффициент = PACF(k)
        try:
            coef, *_ = np.linalg.lstsq(X[:, 1:], X[:, 0], rcond=None)
            pacf[k-1] = coef[-1]
        except Exception:
            pacf[k-1] = 0.0
    # 95% доверительный интервал: ±1.96/sqrt(n)
    ci = 1.96 / math.sqrt(len(stream))
    # порядок памяти = последний лаг, где |PACF| > ci (до того как уходит в шум)
    memory_order = 0
    for k in range(max_lag):
        if abs(pacf[k]) > ci:
            memory_order = k + 1
        elif k > 2:  # допускаем краткий провал
            break
    return {"pacf": pacf.tolist(), "ci95": ci, "memory_order": memory_order,
            "n_glyphs": len(stream)}


# ============================================================
# G3. GENERATIVE SUFFICIENCY
# ============================================================

class NGramGlyphModel:
    """Простейшая слоговая модель: n-грамма глифов. Учится и генерирует."""
    def __init__(self, order: int):
        self.order = order
        self.context_counts = collections.defaultdict(collections.Counter)
        self.vocab = []

    def fit(self, glyph_stream: str):
        self.vocab = sorted(set(glyph_stream))
        o = self.order
        for i in range(len(glyph_stream) - o):
            ctx = glyph_stream[i:i+o]
            nxt = glyph_stream[i+o]
            self.context_counts[ctx][nxt] += 1

    def generate(self, n_chars: int, seed: int = 7) -> str:
        rng = np.random.default_rng(seed)
        starts = list(self.context_counts.keys())
        ctx = starts[rng.integers(len(starts))]
        out = list(ctx)
        for _ in range(n_chars):
            cands = self.context_counts.get(ctx)
            if not cands:
                ctx = starts[rng.integers(len(starts))]; continue
            ws = np.array(list(cands.values()), dtype=float)
            ws /= ws.sum()
            ch = rng.choice(list(cands.keys()), p=ws)
            out.append(ch)
            ctx = "".join(out[-self.order:])
        return "".join(out)


def g3_generative_sufficiency(lines_df) -> dict:
    """
    Выучить 3-граммную модель глифов на реальном Войниче, сгенерировать текст
    той же длины, сравнить ВСЕ ключевые метрики: энтропия H_{2|1}, Ципф-наклон,
    TTR, доля дубликатов биграмм, line-edge профиль (длина строки).
    Если простая модель воспроизводит статистику => она SUFFICIENT (достаточна):
    вся наблюдаемая сложность объясняется короткой слоговой памятью.
    """
    P = lines_df[lines_df["n_tokens"] > 0].copy()
    # реальный поток глифов
    real_stream = ""
    real_words = []
    for toks in P["tokens"]:
        real_words.extend(toks)
        real_stream += ".".join(toks) + "."
    real_stream = real_stream

    # обучим 3-граммную модель
    model = NGramGlyphModel(order=3)
    model.fit(real_stream)
    synth_stream = model.generate(len(real_stream), seed=7)
    # разобьём синтетику на «слова» по точке
    synth_words = [w for w in synth_stream.split(".") if w]

    # метрики для real vs synth
    def metrics(words, stream):
        counts = collections.Counter(words)
        ranks = sorted(counts.values(), reverse=True)
        top = ranks[:50]
        if len(top) >= 5:
            x = np.log10(np.arange(1, len(top)+1))
            y = np.log10(np.array(top, dtype=float))
            zipf_slope = float(np.polyfit(x, y, 1)[0])
        else:
            zipf_slope = float("nan")
        # H_{2|1}
        uni = collections.Counter(stream)
        bi = collections.Counter(stream[i:i+2] for i in range(len(stream)-1))
        H1 = -sum((c/sum(uni.values()))*math.log2(c/sum(uni.values())) for c in uni.values())
        H2 = -sum((c/sum(bi.values()))*math.log2(c/sum(bi.values())) for c in bi.values())
        h21 = H2 - H1
        # dup bigrams
        ng2 = collections.Counter()
        for w in words:
            for a, b in zip(w, w[1:]):
                ng2[(a, b)] += 1
        dup = sum(c for (a, b), c in ng2.items() if a == b)
        tot = sum(ng2.values())
        return {"n_words": len(words), "n_types": len(counts),
                "ttr": len(counts)/len(words) if words else 0,
                "zipf_slope": zipf_slope, "H_2_1": h21,
                "dup_bigram_pct": (dup/tot*100) if tot else 0,
                "mean_wlen": float(np.mean([len(w) for w in words])) if words else 0}

    m_real = metrics(real_words, real_stream)
    m_synth = metrics(synth_words, synth_stream)

    # насколько хорошо воспроизведено (относительная ошибка по каждой метрике)
    repro = {}
    for key in ["ttr", "zipf_slope", "H_2_1", "dup_bigram_pct", "mean_wlen"]:
        r, s = m_real[key], m_synth[key]
        repro[key] = {"real": r, "synth_3gram": s,
                      "rel_err": abs(s - r) / max(abs(r), 1e-9)}
    # общий вердикт: средняя относительная ошибка
    mean_rel_err = float(np.mean([repro[k]["rel_err"] for k in repro]))

    return {"real_metrics": m_real, "synth_3gram_metrics": m_synth,
            "reproduction": repro, "mean_rel_error": mean_rel_err,
            "sample_synthetic": synth_words[:30]}


# ============================================================
# Оркестр
# ============================================================

def run(figures: Path = Path("figures")) -> dict:
    figures.mkdir(exist_ok=True)
    print("=" * 68)
    print("РАУНД 5: текст как физический объект — три новые гипотезы")
    print("=" * 68)
    ld = load_voynich()
    results = {}

    # ---- G1 ----
    print("\n--- G1. WORD-WRAP RECONSTRUCTION ---")
    g1 = g1_word_wrap(ld)
    results["g1_word_wrap"] = {k: v for k, v in g1.items() if k != "line_length_hist"}
    print(f"  Оценка ширины строки: {g1['W_estimated_glyphs']:.1f} глифов (медиана)")
    print(f"  средняя длина строки: {g1['line_length_mean']:.1f} ± {g1['line_length_std']:.1f} (CV={g1['line_length_cv']:.2f})")
    print(f"  корреляция (длина строки, длина последнего слова): {g1['corr_len_lastword']:+.3f}")
    print(f"  F1 восстановления границ переносов: реал={g1['boundary_f1_real']:.3f}  shuffle={g1['boundary_f1_shuffle']:.3f}")
    delta = g1['boundary_f1_real'] - g1['boundary_f1_shuffle']
    print(f"  => модель бьёт shuffle на {delta*100:.0f} пп по точности границ")
    print(f"  => {'текст подчиняется word-wrap (ПИСАЛСЯ под страницу)' if delta > 0.1 else 'word-wrap НЕ объясняет границы — порядок слов не привязан к ширине строки'}")
    # график длин строк
    fig, ax = plt.subplots(figsize=(9, 4))
    hist = np.array(g1["line_length_hist"])
    ax.bar(range(len(hist)), hist, color="#c0392b", alpha=0.8)
    ax.axvline(g1["W_estimated_glyphs"]/2, color="k", ls="--", label=f"оценка ширины ~{g1['W_estimated_glyphs']:.0f}")
    ax.set_xlabel("длина строки (бины)"); ax.set_ylabel("число строк")
    ax.set_title(f"G1: распределение длин строк\nCV={g1['line_length_cv']:.2f} (низкий=писался под страницу)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figures / "R5_G1_wordwrap.png", dpi=130)
    print(f"  -> figures/R5_G1_wordwrap.png")

    # ---- G2 ----
    print("\n--- G2. EFFECTIVE MEMORY LENGTH (PACF) ---")
    g2 = g2_pacf_memory(ld, max_lag=24)
    results["g2_pacf"] = {"memory_order": g2["memory_order"], "ci95": g2["ci95"],
                          "n_glyphs": g2["n_glyphs"]}
    print(f"  порядок памяти процесса: {g2['memory_order']} глифов")
    print(f"  95% CI: ±{g2['ci95']:.4f}")
    print(f"  PACF по лагам: " + " ".join(f"{p:+.2f}" for p in g2["pacf"][:12]))
    print(f"  => память {'КОРОТКАЯ (<5) — слоговая, не синтаксис' if g2['memory_order']<5 else 'ДЛИННАЯ — возможен синтаксис'}")
    fig, ax = plt.subplots(figsize=(9, 4))
    lags = range(1, len(g2["pacf"])+1)
    ax.stem(lags, g2["pacf"], basefmt=" ")
    ax.axhline(g2["ci95"], color="gray", ls=":", label=f"95% CI ±{g2['ci95']:.3f}")
    ax.axhline(-g2["ci95"], color="gray", ls=":")
    ax.axvline(g2["memory_order"]+0.5, color="r", ls="--", label=f"память ≈ {g2['memory_order']}")
    ax.set_xlabel("лаг"); ax.set_ylabel("PACF")
    ax.set_title("G2: partial autocorrelation потока глифов\n(порядок памяти процесса)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figures / "R5_G2_pacf.png", dpi=130)
    print(f"  -> figures/R5_G2_pacf.png")

    # ---- G3 ----
    print("\n--- G3. GENERATIVE SUFFICIENCY (3-граммная модель) ---")
    g3 = g3_generative_sufficiency(ld)
    results["g3_sufficiency"] = {k: g3[k] for k in ["real_metrics", "synth_3gram_metrics",
                                                       "mean_rel_error"]}
    print(f"  метрика             real        3-gram    отн.ошибка")
    for k, v in g3["reproduction"].items():
        print(f"  {k:18s} {v['real']:10.3f}  {v['synth_3gram']:8.3f}  {v['rel_err']*100:7.1f}%")
    print(f"\n  средняя относительная ошибка воспроизведения: {g3['mean_rel_error']*100:.1f}%")
    print(f"  => 3-грамма {'ДОСТАТОЧНА' if g3['mean_rel_error']<0.2 else 'НЕ достаточна'} "
          f"для объяснения статистики Войнича")
    print(f"  пример синтетики: {' '.join(g3['sample_synthetic'][:12])}")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    keys = list(g3["reproduction"].keys())
    real_v = [g3["reproduction"][k]["real"] for k in keys]
    synth_v = [g3["reproduction"][k]["synth_3gram"] for k in keys]
    x = np.arange(len(keys))
    # нормируем для общей шкалы
    def nz(a): a=np.array(a,dtype=float); return a/max(np.abs(a).max(),1e-9)
    ax.bar(x-0.2, nz(real_v), 0.4, label="real Voynich", color="#c0392b")
    ax.bar(x+0.2, nz(synth_v), 0.4, label="3-gram synthetic", color="#2980b9")
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=15)
    ax.set_title(f"G3: 3-граммная модель воспроизводит Войнича?\nсредняя отн.ошибка {g3['mean_rel_error']*100:.1f}%  (нормировано)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figures / "R5_G3_sufficiency.png", dpi=130)
    print(f"  -> figures/R5_G3_sufficiency.png")

    Path("round5_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=lambda o: float(o) if hasattr(o,"__float__") else str(o)),
        encoding="utf-8")
    print(f"\n  -> round5_results.json")
    return results


if __name__ == "__main__":
    run()
