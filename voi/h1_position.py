"""
Гипотеза 1: текст рядом с иллюстрацией описывает иллюстрацию.

Проверки (заявленные числа):
  1. Первые/последние слова строки статистически отличаются от середины:
     JSD ≈ 0.52 (first-vs-rest / last-vs-rest).
  2. Line-edge эффект: суффикс '-am' на конце строки встречается в 12% случаев
     против 0.6% в середине.
  3. Короткие слова тяготеют к началу строки, с определёнными окончаниями - к концу.
  Контроль 1: пересечение слов, типичных для НАЧАЛА АБЗАЦА, со словами,
     типичными для НАЧАЛА СТРОКИ. Из 105 типичных для начала абзаца только 1
     не объясняется тем, что оно короткое и стоит у левого края.
  Контроль 2 (Монте-Карло верх/низ страницы): реальное различие между текстом
     сверху и снизу страницы (JSD 0.81) МЕНЬШЕ, чем при случайной перетасовке
     строк внутри страницы (JSD 0.84) => семантики нет, есть разреженный словарь.

Вывод статьи: гипотеза не подтвердилась. Line-edge эффект реален, но это
вёрстка, а не смысл.
"""
from __future__ import annotations
import collections
import numpy as np
from voi import common


# ---- позиция слова в строке ----

def first_mid_last_word_tokens(lines_df):
    """
    По каждой параграфной строке (P-локус): первое слово, последнее слово,
    середина. Возвращает три списка токенов (с повторами).
    """
    firsts, lasts, mids = [], [], []
    for toks, ltype in zip(lines_df["tokens"], lines_df["locus_type"]):
        if not ltype.startswith("P") or len(toks) == 0:
            continue
        if len(toks) == 1:
            firsts.append(toks[0])
            continue
        firsts.append(toks[0])
        lasts.append(toks[-1])
        mids.extend(toks[1:-1])
    return firsts, lasts, mids


def line_edge_jsd(lines_df, against="rest"):
    """
    JSD между распределением слов на краю строки и остальным текстом.
    against='rest': край vs всё остальное (как в статье ~0.52).
    Возвращает (jsd_first, jsd_last).
    """
    firsts, lasts, mids = first_mid_last_word_tokens(lines_df)
    alltok = firsts + lasts + mids
    p_all = common.freq_dist(alltok)
    p_first = common.freq_dist(firsts)
    p_last = common.freq_dist(lasts)
    p_mid = common.freq_dist(mids)
    if against == "rest":
        jsd_first = common.js_divergence(p_first, p_mid)
        jsd_last = common.js_divergence(p_last, p_mid)
    else:  # against corpus
        jsd_first = common.js_divergence(p_first, p_all)
        jsd_last = common.js_divergence(p_last, p_all)
    return jsd_first, jsd_last, len(firsts), len(lasts), len(mids)


def suffix_am_line_edge(lines_df):
    """
    Доля слов с суффиксом '-am' на конце строки vs в середине.
    Также общий тест: доля слов, оканчивающихся на каждый суффикс, на краю vs середине.
    """
    firsts, lasts, mids = first_mid_last_word_tokens(lines_df)

    def frac_suffix(words, suf):
        n = sum(1 for w in words if w.endswith(suf))
        return n / len(words) if words else 0.0

    am_last = frac_suffix(lasts, "am")
    am_mid = frac_suffix(mids, "am")
    am_first = frac_suffix(firsts, "am")
    return {"am_last": am_last, "am_mid": am_mid, "am_first": am_first,
            "ratio_last_mid": am_last / max(am_mid, 1e-9)}


# ---- Контроль 1: начало абзаца vs начало строки ----

def paragraph_start_vs_line_start(lines_df, top_k=105):
    """
    Статья: слова, характерные для начала АБЗАЦА vs начала СТРОКИ.
    Логика автора: если эффект начала абзаца имеет СМЫСЛОВУЮ природу, он должен
    быть независим от эффекта начала строки. Если же обе позиции совпадают по
    составу характерных слов, эффект начала абзаца сводится к механике строки
    (короткое слово / line-start маркер оказывается у левого края).

    Здесь:
      para_first[w]  - сколько раз w первым в абзаце
      line_first[w]  - сколько раз w первым в строке
      word_total[w]  - общая частота w
      line_start_ratio[w] = line_first[w]/word_total[w] - доля случаев, когда w
                            стоит в начале строки (характерность левого края)

    'Объясняется левым краем' = line_start_ratio высокий (>= 0.10) и w достаточно
    частое (>= 3), т.е. это типичный line-start маркер. Подсчёт того, сколько
    из top-K para-start слов объясняются этим, повторяет постановку статьи:
    "из 105 слов ... только одно не объясняется".
    """
    para_first = collections.Counter()
    line_first = collections.Counter()
    word_total = collections.Counter()
    word_len = {}

    for toks, ltype, pstart in zip(lines_df["tokens"], lines_df["locus_type"],
                                   lines_df["para_start"]):
        if not ltype.startswith("P") or len(toks) == 0:
            continue
        line_first[toks[0]] += 1
        if pstart:
            para_first[toks[0]] += 1
        for t in toks:
            word_total[t] += 1
            word_len[t] = len(t)

    top_para = [w for w, _ in para_first.most_common(top_k)]

    # характерные line-start слова
    line_start_markers = set()
    for w, c in word_total.items():
        if c >= 3 and (line_first[w] / c) >= 0.10:
            line_start_markers.add(w)

    n_unexplained = 0
    unexplained = []
    for w in top_para:
        explained = (w in line_start_markers) or (word_len[w] <= 4 and word_total[w] >= 5)
        if not explained:
            n_unexplained += 1
            unexplained.append(w)
    return {
        "top_para_n": len(top_para),
        "n_unexplained": n_unexplained,
        "unexplained_examples": unexplained[:15],
        "median_word_len": float(np.median([word_len[w] for w in word_len])),
        "para_first_total": sum(para_first.values()),
        "line_first_total": sum(line_first.values()),
        "n_line_start_markers": len(line_start_markers),
        "intersection_with_linefirst": len(set(top_para) & line_start_markers),
    }


# ---- Контроль 2: Монте-Карло верх/низ страницы ----

def _topbottom_jsd_perpage(page_to_lines, pages):
    """
    Средний по страницам JSD между верхними и нижними половинами строк страницы.
    (Per-page агрегация естественнее для 'различия текста сверху/снизу страницы',
    чем pooled по всему корпусу.)
    """
    js = []
    for f in pages:
        ls = page_to_lines[f]
        half = len(ls) // 2
        top = [t for row in ls[:half] for t in row]
        bot = [t for row in ls[half:] for t in row]
        if top and bot:
            js.append(common.js_divergence(common.freq_dist(top), common.freq_dist(bot)))
    return float(np.mean(js)) if js else 0.0


def mc_top_bottom(lines_df, n_iter=1000, seed=42):
    """
    Реальный JSD(верх vs низ страницы) [per-page mean] vs распределение JSD
    при случайной перетасовке строк внутри каждой страницы.
    Статья: реальный 0.81, перетасовка 0.84 -> различие пренебрежимо мало.
    """
    rng = np.random.default_rng(seed)
    page_lines = collections.defaultdict(list)
    for folio, toks in zip(lines_df["folio"], lines_df["tokens"]):
        if toks:
            page_lines[folio].append(list(toks))
    pages = [f for f, ls in page_lines.items() if len(ls) >= 4]

    real_jsd = _topbottom_jsd_perpage(page_lines, pages)

    null = np.empty(n_iter)
    for i in range(n_iter):
        shuffled = {}
        for f in pages:
            ls = [list(x) for x in page_lines[f]]
            rng.shuffle(ls)
            shuffled[f] = ls
        null[i] = _topbottom_jsd_perpage(shuffled, pages)
    p = float(np.mean(null >= real_jsd))
    return {
        "real_jsd": real_jsd,
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null)),
        "p_real_ge_null": p,
        "n_pages": len(pages),
        "n_iter": n_iter,
    }


def run(lines_df, verbose=True) -> dict:
    # только параграфный текст
    P = lines_df[lines_df["locus_type"].str.startswith("P") & (lines_df["tokens"].apply(len) > 0)].copy()
    out = {}

    # 1. line-edge JSD
    jsd_f, jsd_l, nf, nl, nm = line_edge_jsd(P)
    out["jsd_first_vs_mid"] = jsd_f
    out["jsd_last_vs_mid"] = jsd_l
    out["n_first"] = nf
    out["n_last"] = nl
    out["n_mid"] = nm
    if verbose:
        print("=== ГИПОТЕЗА 1: позиция в абзаце ===")
        print(f"JSD(первое слово строки vs середина) = {jsd_f:.3f}")
        print(f"JSD(последнее слово строки vs середина) = {jsd_l:.3f}")
        print(f"  (статья: ~0.52 для краёв vs середины)")
        print()

    # 2. суффикс -am
    am = suffix_am_line_edge(P)
    out["suffix_am"] = am
    if verbose:
        print(f"Суффикс '-am': на конце строки {am['am_last']*100:.1f}% vs середина {am['am_mid']*100:.2f}%")
        print(f"  (статья: 12% на конце vs 0.6% в середине), отношение {am['ratio_last_mid']:.1f}×")
        print()

    # 3. контроль 1: начало абзаца vs начало строки
    c1 = paragraph_start_vs_line_start(P, top_k=105)
    out["control1"] = c1
    if verbose:
        print(f"Контроль 1 (начало абзаца vs начало строки):")
        print(f"  типичных для начала абзаца (top-105): {c1['top_para_n']}")
        print(f"  не объясняются короткостью: {c1['n_unexplained']}")
        print(f"  (статья: из 105 только 1 не объясняется)")
        print(f"  примеры 'необъяснимых': {c1['unexplained_examples']}")
        print(f"  пересечение с типичными для начала строки: {c1['intersection_with_linefirst']}")
        print()

    # 4. контроль 2: Монте-Карло верх/низ страницы
    c2 = mc_top_bottom(P, n_iter=1000)
    out["control2"] = c2
    if verbose:
        print(f"Контроль 2 (Монте-Карло верх/низ страницы):")
        print(f"  реальный JSD(верх vs низ) = {c2['real_jsd']:.3f}")
        print(f"  null (перетасовка строк) mean = {c2['null_mean']:.3f} ± {c2['null_std']:.3f}")
        print(f"  P(null >= real) = {c2['p_real_ge_null']:.3f}")
        print(f"  (статья: реальный 0.81 < перетасовка 0.84)")
        print()
    return out


if __name__ == "__main__":
    from voi.parse_ivtff import parse_ivtff, add_n_token_columns
    ld, _ = parse_ivtff(str(common.data_path("voynich")))
    ld = add_n_token_columns(ld)
    run(ld)
