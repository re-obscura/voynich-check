"""
Раунд 6: глубже — где Войнич ЛОМАЕТ свою собственную модель?

После G3 (раунд 5) мы знаем: 3-грамма глифов воспроизводит статистику с ошибкой 0.9%.
Новый вопрос: где она НЕ воспроизводит? Любой остаток (residual) — кандидат на
спрятанный смысл, потому что смысл = то, что не выводится из локальной статистики.

H1. RESIDUAL MAP — тепловая карта непредсказуемости по фолио. Какие фолио/секции
    «труднее» всего для модели? Если остаток распределён случайно — смысла нет.
    Если сконцентрирован (напр. в bio-B) — это сигнал.

H2. СЛОГОВАЯ ГРАММАТИКА СЛОВА — явная реконструкция слотов (префикс/корень/суффикс)
    через позиционные распределения глифов, проверка консистентности.

H3. bio-B ПРИЦЕЛЬНАЯ АТАКА — сравнить остаток 3-граммы на bio-B vs herbal-B.
    bio-B — единственная необъяснённая аномалия за 5 раундов.

H4. INFORMATION BOUND — сколько бит «смысла» максимально спрятано в остатке?
    Разность между реальной условной энтропией и энтропией нашей модели = верхняя
    граница информации, не объяснимой локальной статистикой.

H5. TOPIC SIGNATURES — слова, повторяющиеся на конкретных фолио сверх 3-граммного
    ожидания. Кандидаты на «названия» (если есть meaning, они здесь).
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
from voynich_lm.analyze.round5 import NGramGlyphModel


def load_voynich():
    ld, _ = parse_ivtff(str(data_path("voynich")))
    ld = add_n_token_columns(ld)
    return ld


# ============================================================
# Подготовка: 3-граммная модель + per-position surprisal
# ============================================================

def build_model_and_stream(lines_df, order: int = 3):
    P = lines_df[lines_df["n_tokens"] > 0].copy()
    # глобальный поток глифов с маркерами строк
    stream = ""
    word_starts = []  # (char_idx_in_stream, word) для привязки к фолио
    folio_of_pos = []
    cur_pos = 0
    for f, toks, cur, sec in zip(P["folio"], P["tokens"], P["currier"], P["section"]):
        for w in toks:
            for c in w:
                stream += c; folio_of_pos.append(f)
            stream += "."; folio_of_pos.append(f)  # разделитель в пределах слова/строки
    # обучим модель
    model = NGramGlyphModel(order=order)
    model.fit(stream)
    return model, stream, folio_of_pos


def per_position_surprisal(model, stream):
    """-log2 P(glyph_i | context) для каждой позиции (бит). Мера «непредсказуемости»."""
    order = model.order
    surprisal = np.zeros(len(stream))
    for i in range(order, len(stream)):
        ctx = stream[i-order:i]
        cands = model.context_counts.get(ctx)
        if not cands:
            surprisal[i] = math.log2(len(model.vocab))  # равномерное по умолчанию
            continue
        total = sum(cands.values())
        ch = stream[i]
        p = (cands.get(ch, 0) + 0.01) / (total + 0.01*len(model.vocab))  # add-smoothing
        surprisal[i] = -math.log2(p)
    return surprisal


# ============================================================
# H1. RESIDUAL MAP по фолио
# ============================================================

def h1_residual_map(surprisal, folio_of_pos, ld) -> dict:
    """Средний surprisal по фолио -> какие «труднее» для модели?"""
    by_folio = collections.defaultdict(list)
    for pos, f in enumerate(folio_of_pos):
        if pos < len(surprisal):
            by_folio[f].append(surprisal[pos])
    # агрегат
    folio_surp = {f: float(np.mean(v)) for f, v in by_folio.items() if len(v) > 20}
    # по секции/диалекту
    meta = {}
    for f, row in zip(ld["folio"], ld[["section","currier"]].itertuples(index=False)):
        meta.setdefault(f, (row.section, row.currier))
    by_group = collections.defaultdict(list)
    for f, s in folio_surp.items():
        sec, cur = meta.get(f, ("?","?"))
        by_group[f"{sec}/{cur}"].append(s)
    group_stats = {g: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                   for g, v in by_group.items()}
    # глобальный разброс — значим ли он?
    all_vals = list(folio_surp.values())
    global_mean, global_std = float(np.mean(all_vals)), float(np.std(all_vals))
    # сравнение: разброс между фолио vs разброс внутри (ANOVA-like)
    within = []
    for f, v in by_folio.items():
        if len(v) > 20:
            within.append(float(np.std(v)))
    within_mean = float(np.mean(within))
    return {"folio_surprisal": folio_surp, "by_group": group_stats,
            "global_mean": global_mean, "global_std": global_std,
            "within_folio_std_mean": within_mean,
            "between_over_within": global_std / within_mean if within_mean > 0 else 0}


# ============================================================
# H2. СЛОГОВАЯ ГРАММАТИКА СЛОВА
# ============================================================

def h2_syllabic_grammar(lines_df) -> dict:
    """
    Реконструкция слотов через позиционные распределения глифов.
    Для каждой позиции в слове (0..max) — частотный профиль глифов.
    Если есть стабильные слоты (напр. позиция 0 всегда {o,a,k,t,q,p}),
    это явная слоговая грамматика.
    """
    P = lines_df[lines_df["n_tokens"] > 0].copy()
    # позиционные частоты глифов
    pos_freq = collections.defaultdict(collections.Counter)
    pos_total = collections.Counter()
    for toks in P["tokens"]:
        for w in toks:
            for i, c in enumerate(w):
                if i >= 8: continue  # только первые 8 позиций
                pos_freq[i][c] += 1
                pos_total[i] += 1
    # энтропия распределения глифов по позициям (низкая = чёткий слот)
    pos_entropy = {}
    top_glyhs_per_pos = {}
    for i in range(8):
        if pos_total[i] == 0: continue
        freqs = {c: cnt/pos_total[i] for c, cnt in pos_freq[i].items()}
        H = -sum(p*math.log2(p) for p in freqs.values() if p > 0)
        pos_entropy[i] = H
        top = sorted(freqs.items(), key=lambda x: -x[1])[:5]
        top_glyhs_per_pos[i] = [(c, round(p,3)) for c, p in top]
    # реконструкция «слотов»: какие глифы характерны для начала/середины/конца
    return {"pos_entropy": pos_entropy, "top_glyphs_per_pos": top_glyhs_per_pos,
            "pos_total": dict(pos_total)}


# ============================================================
# H3. bio-B ПРИЦЕЛЬНАЯ АТАКА
# ============================================================

def h3_bio_b_attack(surprisal, folio_of_pos, ld) -> dict:
    """Сравнить остаток (surprisal) на bio-B vs herbal-B vs recipes-B."""
    meta = {}
    for f, row in zip(ld["folio"], ld[["section","currier"]].itertuples(index=False)):
        meta[f] = (row.section, row.currier)
    by_group = collections.defaultdict(list)
    for pos, f in enumerate(folio_of_pos):
        if pos >= len(surprisal): continue
        sec, cur = meta.get(f, ("?","?"))
        if cur == "B":
            by_group[sec].append(surprisal[pos])
    # также A-herbal для референса
    for pos, f in enumerate(folio_of_pos):
        if pos >= len(surprisal): continue
        sec, cur = meta.get(f, ("?","?"))
        if cur == "A" and sec == "herbal":
            by_group["herbal-A"].append(surprisal[pos])
    stats = {g: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
             for g, v in by_group.items()}
    return stats


# ============================================================
# H4. INFORMATION BOUND
# ============================================================

def h4_information_bound(lines_df) -> dict:
    """
    Верхняя граница «смысла» в остатке.
    Истинная энтропийная ставка H* (которую мы не знаем) <= то, что выучила LM.
    Но мы можем оценить НИЖНЮЮ границу того, сколько информации НЕ объясняется
    3-граммой: разность между H* (оценённой через LM с длинным контекстом из
    ARTICLE_LM) и H_3граммы.

    Если LM (ctx=256) даёт CE 1.77 бит/глиф, а 3-грамма — H_3 ≈ 2.0,
    то разность ~0.2 бит/глиф — это вклад ДЛИННОГО контекста. Если смысл
    есть, он ограничен этой разностью. Это верхняя граница информации,
    доступной «смыслу», не объяснимой локальной слоговой моделью.
    """
    P = lines_df[lines_df["n_tokens"] > 0].copy()
    stream = ""
    for toks in P["tokens"]:
        stream += ".".join(toks) + "."
    # H_3граммы (кросс-энтропия модели на своём же тексте)
    model = NGramGlyphModel(order=3); model.fit(stream)
    order = 3
    ce_3gram = 0.0; n = 0
    for i in range(order, len(stream)):
        ctx = stream[i-order:i]
        cands = model.context_counts.get(ctx)
        if not cands: continue
        total = sum(cands.values())
        p = (cands.get(stream[i],0)+0.01)/(total+0.01*len(model.vocab))
        ce_3gram += -math.log2(p); n += 1
    ce_3gram /= n
    # LM (из ARTICLE_LM) дала CE ctx=256 = 1.771 бит/глиф
    ce_lm_256 = 1.771
    # информация, не объяснимая 3-граммой (вклад длинного контекста)
    long_context_info = ce_3gram - ce_lm_256
    return {"H_3gram_bits": ce_3gram, "CE_LM_ctx256_bits": ce_lm_256,
            "long_context_information": long_context_info,
            "pct_unexplained": long_context_info / ce_3gram * 100}


# ============================================================
# H5. TOPIC SIGNATURES
# ============================================================

def h5_topic_signatures(lines_df) -> dict:
    """
    Слова, сверх-частые на конкретных фолио относительно глобальной частоты.
    Кандидаты на «названия» (если есть meaning, подпись растения = слово,
    повторяющееся именно на этом фолио).
    Мера: для каждого слова w и фолио f, (count(w in f)+α)/(N_f) vs глобальная P(w).
    Если локальная частота >> глобальной — сигнатура.
    """
    P = lines_df[lines_df["n_tokens"] > 0].copy()
    # глобальные частоты слов
    global_counts = collections.Counter()
    folio_counts = collections.defaultdict(collections.Counter)
    folio_total = collections.Counter()
    for f, toks in zip(P["folio"], P["tokens"]):
        for w in toks:
            global_counts[w] += 1
            folio_counts[f][w] += 1
            folio_total[f] += 1
    G = sum(global_counts.values())
    global_p = {w: c/G for w, c in global_counts.items()}
    # для каждого фолио — топ-сигнатуры (log локальная/глобальная)
    signatures = {}
    alpha = 1.0
    for f in folio_counts:
        Nf = folio_total[f]
        sigs = []
        for w, c in folio_counts[f].items():
            local_p = (c + alpha) / (Nf + alpha*len(global_p))
            ratio = local_p / global_p.get(w, alpha/G)
            if c >= 3 and ratio > 3:  # минимум 3 раза и в 3× чаще глобальной
                sigs.append((w, ratio, c))
        sigs.sort(key=lambda x: -x[1])
        if sigs:
            signatures[f] = sigs[:5]
    # сколько фолио имеют «сильные» сигнатуры?
    n_with_sig = sum(1 for s in signatures.values() if len(s) >= 3)
    return {"n_folios_with_signatures": n_with_sig, "n_folios_total": len(folio_counts),
            "example_signatures": dict(list(signatures.items())[:8])}


# ============================================================
# Оркестр
# ============================================================

def run(figures: Path = Path("figures")) -> dict:
    figures.mkdir(exist_ok=True)
    print("=" * 68)
    print("РАУНД 6: глубже — где Войнич ломает свою модель?")
    print("=" * 68)
    ld = load_voynich()
    model, stream, folio_of_pos = build_model_and_stream(ld, order=3)
    surprisal = per_position_surprisal(model, stream)
    print(f"  3-граммная модель обучена, surprisal посчитан для {len(stream)} позиций")
    print(f"  средний surprisal: {np.mean(surprisal[3:]):.3f} бит/глиф")
    results = {}

    # ---- H1 ----
    print("\n--- H1. RESIDUAL MAP по фолио ---")
    h1 = h1_residual_map(surprisal, folio_of_pos, ld)
    results["h1_residual_map"] = {k: h1[k] for k in ["global_mean","global_std",
            "within_folio_std_mean","between_over_within","by_group"]}
    print(f"  средний surprisal: {h1['global_mean']:.3f} ± {h1['global_std']:.3f} (между фолио)")
    print(f"  разброс внутри фолио (средний): {h1['within_folio_std_mean']:.3f}")
    print(f"  отношение между/внутри: {h1['between_over_within']:.3f}")
    print(f"  => {'разброс между фолио ЗНАЧИМ (есть «трудные» фолио)' if h1['between_over_within']>0.3 else 'разброс случайный — все фолио одинаково предсказуемы'}")
    print(f"\n  по группам секция/диалект:")
    for g, s in sorted(h1['by_group'].items(), key=lambda x: x[1]['mean']):
        print(f"    {g:18s}: surprisal={s['mean']:.3f} ± {s['std']:.3f} (n={s['n']})")

    # ---- H2 ----
    print("\n--- H2. СЛОГОВАЯ ГРАММАТИКА СЛОВА ---")
    h2 = h2_syllabic_grammar(ld)
    results["h2_syllabic"] = {"pos_entropy": h2["pos_entropy"],
                              "top_glyphs_per_pos": h2["top_glyphs_per_pos"]}
    print(f"  позиционная энтропия глифов:")
    for i in range(8):
        if i in h2["pos_entropy"]:
            top = " ".join(f"{c}({p})" for c,p in h2["top_glyphs_per_pos"][i])
            print(f"    поз.{i}: H={h2['pos_entropy'][i]:.2f}  топ: {top}")

    # ---- H3 ----
    print("\n--- H3. bio-B ПРИЦЕЛЬНАЯ АТАКА (surprisal остаток) ---")
    h3 = h3_bio_b_attack(surprisal, folio_of_pos, ld)
    results["h3_bio_b"] = h3
    print(f"  surprisal по секциям внутри Currier B:")
    for g, s in sorted(h3.items(), key=lambda x: x[1]['mean']):
        print(f"    {g:12s}: {s['mean']:.3f} ± {s['std']:.3f} (n={s['n']})")

    # ---- H4 ----
    print("\n--- H4. INFORMATION BOUND (сколько бит «смысла» возможно?) ---")
    h4 = h4_information_bound(ld)
    results["h4_info_bound"] = h4
    print(f"  H_3граммы (локальная модель):   {h4['H_3gram_bits']:.3f} бит/глиф")
    print(f"  CE LM ctx=256 (длинный контекст): {h4['CE_LM_ctx256_bits']:.3f} бит/глиф")
    print(f"  информация длинного контекста:    {h4['long_context_information']:.3f} бит/глиф ({h4['pct_unexplained']:.1f}%)")
    print(f"  => максимум «смысла», не объяснимого локальной моделью: {h4['long_context_information']:.2f} бит/глиф")

    # ---- H5 ----
    print("\n--- H5. TOPIC SIGNATURES (слова-подписи фолио) ---")
    h5 = h5_topic_signatures(ld)
    results["h5_signatures"] = {"n_with_sig": h5["n_folios_with_signatures"],
                                 "n_total": h5["n_folios_total"]}
    print(f"  фолио с сильными сигнатурами (>=3 слов в 3× чаще): {h5['n_folios_with_signatures']}/{h5['n_folios_total']}")
    print(f"  примеры подписей:")
    for f, sigs in list(h5["example_signatures"].items())[:6]:
        top = " ".join(f"{w}({c}×,{r:.1f})" for w,r,c in sigs[:3])
        print(f"    {f}: {top}")

    # ---- графики ----
    # H1 residual heatmap по фолио (в порядке folio_num)
    fig, ax = plt.subplots(figsize=(13, 3.5))
    fs = sorted(h1["folio_surprisal"].keys(), key=lambda f: (int(re.match(r'f(\d+)',f).group(1)), f))
    vals = [h1["folio_surprisal"][f] for f in fs]
    colors_map = []
    meta = {}
    for f, row in zip(ld["folio"], ld[["section","currier"]].itertuples(index=False)):
        meta[f] = (row.section, row.currier)
    cur_c = {"A":"#2980b9","B":"#c0392b","?":"#7f8c8d"}
    colors_map = [cur_c.get(meta.get(f,("?","?"))[1],"#999") for f in fs]
    ax.scatter(range(len(fs)), vals, c=colors_map, s=20, alpha=0.8)
    ax.axhline(h1["global_mean"], color="k", ls="--", lw=1, label=f"mean {h1['global_mean']:.2f}")
    ax.set_xlabel("фолио (в порядке)"); ax.set_ylabel("surprisal (бит/глиф)")
    ax.set_title("H1: residual map — непредсказуемость по фолио\n(синий=Currier A, красный=B)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figures / "R6_H1_residual.png", dpi=130)
    print(f"\n  -> figures/R6_H1_residual.png")

    # H2 позиционная энтропия
    fig, ax = plt.subplots(figsize=(8, 4))
    pos = sorted(h2["pos_entropy"].keys())
    ax.bar(pos, [h2["pos_entropy"][p] for p in pos], color="#27ae60")
    ax.set_xlabel("позиция в слове"); ax.set_ylabel("энтропия глифов (бит)")
    ax.set_title("H2: слоговая грамматика — энтропия слотов по позициям\n(низкая = чёткий слот)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(figures / "R6_H2_syllabic.png", dpi=130)
    print(f"  -> figures/R6_H2_syllabic.png")

    Path("round6_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=lambda o: float(o) if hasattr(o,"__float__") else str(o)),
        encoding="utf-8")
    print(f"  -> round6_results.json")
    return results


if __name__ == "__main__":
    run()
