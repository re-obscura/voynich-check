"""
Гипотеза 5: самоцитирование по методу Тимма и Шиннера.

Timm & Schinner (2019, Cryptologia): писец брал одну/несколько предыдущих
строк и модифицировал их (префиксы/суффиксы), получая новую строку -> self-citation.

Если так, слова в строке должны ЗАМЕТНО чаще повторять слова из нескольких
предыдущих строк, чем в естественном тексте.

Self-citation rate (по статье):
  Для каждой строки i и окна N предыдущих строк: доля слов строки i,
  которые встречаются хотя бы в одной из N предыдущих строк.
  Сравнение с Монте-Карло baseline (перемешивание строк внутри страницы)
  и с 4 контрольными текстами (KJV, Moby Dick, Paradise Lost, Julius Caesar).

Заявленные числа:
  - эффект есть и значим: z=9.2, p<0.0001
  - НО эффект 1.12x к случайному фону - СЛАБЕЕ, чем во всех 4 контрольных
    текстах (1.30x-1.68x). Это противоречит модели построчного копирования.
  - Hapax legomena (слова, встречающиеся 1 раз: 6152 из 8000 уникальных)
    дают ровно НУЛЕВОЙ self-citation.
  - Похожесть соседних строк выше случайных пар (Манн-Уитни, p=0.0002),
    но разница крошечная - 0.4% от типичного разброса JSD.

Вывод статьи: эффект реален, но слабее естественных языков -> модель
Тимма/Шиннера (построчное копирование) не подтверждается; всё объясняется
частыми словами.
"""
from __future__ import annotations
import collections
import re
import numpy as np
from scipy import stats

from voi import common


# ---------- self-citation rate ----------

def self_citation_rate(lines_tokens, window=1, restrict_to=None):
    """
    lines_tokens: список списков токенов (строк) в порядке.
    window: сколько предыдущих строк учитывать (N).
    restrict_to: если задано множество слов - считать только их.

    Для каждой строки i>0: доля её слов, которые присутствуют в объединении
    token-множеств строк [i-window .. i-1].
    Возвращает среднюю долю по всем строкам (длины >=1).
    """
    if len(lines_tokens) < 2:
        return 0.0
    # префиксные объединения множеств
    rates = []
    for i in range(1, len(lines_tokens)):
        if window == 0:
            prev_set = set()
        else:
            lo = max(0, i - window)
            prev_set = set()
            for j in range(lo, i):
                prev_set |= set(lines_tokens[j])
        cur = lines_tokens[i]
        if restrict_to is not None:
            cur = [t for t in cur if t in restrict_to]
        if not cur:
            continue
        hit = sum(1 for t in cur if t in prev_set)
        rates.append(hit / len(cur))
    return float(np.mean(rates)) if rates else 0.0


def mc_baseline(lines_tokens, window, n_iter=200, seed=42, by_page=None):
    """
    Монте-Карло: перемешать строки (внутри страницы, если by_page задан) и
    пересчитать self-citation rate.
    Возвращает средний baseline и std.
    """
    rng = np.random.default_rng(seed)
    vals = []
    if by_page:
        groups = collections.defaultdict(list)
        for folio, toks in by_page:
            groups[folio].append(toks)
        folios = list(groups.keys())
    for _ in range(n_iter):
        if by_page:
            new = []
            for f in folios:
                g = [list(x) for x in groups[f]]
                rng.shuffle(g)
                new.extend(g)
        else:
            new = [list(x) for x in lines_tokens]
            rng.shuffle(new)
        vals.append(self_citation_rate(new, window=window))
    return float(np.mean(vals)), float(np.std(vals))


# ---------- JSD похожесть соседних строк ----------

def neighbor_jsd_distribution(lines_tokens, window=1):
    """
    Для каждой пары (строка i, строка i-window): JSD их частотных распределений.
    Возвращает массив JSD-значений.
    """
    jsds = []
    for i in range(window, len(lines_tokens)):
        a = lines_tokens[i]
        b = lines_tokens[i - window]
        if not a or not b:
            continue
        jsds.append(common.js_divergence(common.freq_dist(a), common.freq_dist(b)))
    return np.array(jsds)


def random_pair_jsd_distribution(lines_tokens, n_pairs=2000, seed=42):
    """JSD между случайными парами строк."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(lines_tokens))
    jsds = []
    for _ in range(n_pairs):
        a, b = rng.choice(idx, 2, replace=False)
        ta, tb = lines_tokens[a], lines_tokens[b]
        if not ta or not tb:
            continue
        jsds.append(common.js_divergence(common.freq_dist(ta), common.freq_dist(tb)))
    return np.array(jsds)


# ---------- контрольные тексты ----------

def load_gutenberg(path, max_words=None):
    """Прочитать Project Gutenberg текст, вырезать хедер/футер, токенизировать."""
    with open(path, encoding="utf-8", errors="ignore") as fh:
        txt = fh.read()
    start = txt.find("*** START OF")
    end = txt.find("*** END OF")
    if start != -1 and end != -1:
        txt = txt[start:end]
    # токенизация на слова (нижний регистр)
    words = re.findall(r"[A-Za-z']+", txt.lower())
    if max_words:
        words = words[:max_words]
    return words


def words_to_lines(words, line_len=8):
    """
    Разбить поток слов на 'строки' фиксированной длины (как строки рукописи).
    line_len ~ средняя длина строки Войнича (~8 слов).
    """
    return [words[i:i + line_len] for i in range(0, len(words) - line_len + 1, line_len)]


def cap_to_voynich_size(lines_tokens, target_lines):
    """Ограничить число строк для скорости (берём первые target_lines)."""
    if len(lines_tokens) > target_lines:
        return lines_tokens[:target_lines]
    return lines_tokens


def run(lines_df, verbose=True) -> dict:
    out = {}
    # параграфный текст в порядке следования (по фолио, по locus_seq)
    P = lines_df[lines_df["locus_type"].str.startswith("P") & (lines_df["tokens"].apply(len) > 0)].copy()
    P = P.sort_values(["folio_num", "locus_seq"]).reset_index(drop=True)
    lines_tokens = [list(t) for t in P["tokens"]]
    by_page = list(zip(P["folio"], lines_tokens))
    n_lines = len(lines_tokens)

    if verbose:
        print("=== ГИПОТЕЗА 5: самоцитирование (Тимм/Шиннер) ===")
        print(f"строк в анализе: {n_lines}")

    alltok = [t for ts in lines_tokens for t in ts]
    freq = collections.Counter(alltok)
    hapax = {w for w, c in freq.items() if c == 1}
    out["n_unique"] = len(freq)
    out["n_hapax"] = len(hapax)
    if verbose:
        print(f"уникальных слов: {len(freq)} (статья ~8000)")
        print(f"hapax legomena: {len(hapax)} (статья 6152)")
        print()

    # 1. self-citation rate vs MC, для нескольких окон
    scr_results = {}
    for N in (1, 2, 3, 5, 10):
        real = self_citation_rate(lines_tokens, window=N)
        base_mean, base_std = mc_baseline(lines_tokens, window=N, n_iter=200, by_page=by_page)
        if base_std > 0:
            z = (real - base_mean) / base_std
        else:
            z = float("nan")
        ratio = real / base_mean if base_mean > 0 else float("nan")
        scr_results[N] = {"real": real, "null_mean": base_mean, "null_std": base_std,
                          "z": z, "ratio": ratio}
        if verbose:
            print(f"окно N={N:2d}: real={real:.4f}  null={base_mean:.4f}±{base_std:.4f}  z={z:+.1f}  ratio={ratio:.3f}x")
    out["scr_by_window"] = scr_results
    if verbose:
        print(f"  (статья: эффект значим z=9.2; ratio ~1.12x)")
        print()

    # 2. hapax self-citation
    hapax_real = self_citation_rate(lines_tokens, window=1, restrict_to=hapax)
    out["hapax_self_cite"] = hapax_real
    if verbose:
        print(f"Self-citation ТОЛЬКО для hapax (окно 1): {hapax_real:.6f}")
        print(f"  (статья: hapax дают ровно 0)")
        print()

    # 3. контрольные тексты
    controls = {}
    target = min(n_lines, 4000)
    for name, path in [("KJV (Библия)", str(common.data_path("kjv"))),
                       ("Moby Dick", str(common.data_path("moby"))),
                       ("Paradise Lost", str(common.data_path("paradise"))),
                       ("Julius Caesar", str(common.data_path("caesar")))]:
        try:
            words = load_gutenberg(path, max_words=40000)
            ctl_lines = words_to_lines(words, line_len=8)
            ctl_lines = cap_to_voynich_size(ctl_lines, target)
            real = self_citation_rate(ctl_lines, window=1)
            base_mean, base_std = mc_baseline(ctl_lines, window=1, n_iter=100, by_page=None)
            ratio = real / base_mean if base_mean > 0 else float("nan")
            controls[name] = {"n_lines": len(ctl_lines), "real": real,
                              "null_mean": base_mean, "ratio": ratio}
            if verbose:
                print(f"контроль [{name}]: real={real:.4f}  null={base_mean:.4f}  ratio={ratio:.3f}x")
        except FileNotFoundError:
            if verbose:
                print(f"контроль [{name}]: файл не найден")
    out["controls"] = controls
    voyn_ratio = scr_results[1]["ratio"]
    if verbose:
        print(f"  Войнич ratio (N=1): {voyn_ratio:.3f}x")
        print(f"  (статья: Войнич 1.12x - НИЖЕ всех контрольных 1.30-1.68x)")
        # проверим, ниже ли всех
        ctl_ratios = [c["ratio"] for c in controls.values()]
        if ctl_ratios:
            below_all = voyn_ratio < min(ctl_ratios)
            print(f"  Войнич ниже всех контрольных? {below_all}  (контрольные ratios: {[round(r,2) for r in ctl_ratios]})")
        print()

    # 4. JSD соседних vs случайных (Манн-Уитни)
    nb = neighbor_jsd_distribution(lines_tokens, window=1)
    rnd = random_pair_jsd_distribution(lines_tokens, n_pairs=2000)
    mw = stats.mannwhitneyu(nb, rnd, alternative="less")
    diff_frac = (np.mean(rnd) - np.mean(nb)) / np.std(rnd) if np.std(rnd) > 0 else float("nan")
    out["neighbor_jsd"] = {"neighbor_mean": float(np.mean(nb)), "random_mean": float(np.mean(rnd)),
                           "mw_p": float(mw.pvalue), "diff_frac_of_std": float(diff_frac)}
    if verbose:
        print(f"JSD соседних строк: mean={np.mean(nb):.4f}, случайные пары: mean={np.mean(rnd):.4f}")
        print(f"Манн-Уитни (соседние < случайных) p={mw.pvalue:.4f}")
        print(f"разница как доля std(случайных): {diff_frac*100:.2f}%")
        print(f"  (статья: p=0.0002, разница 0.4% от разброса)")
        print()

    return out


if __name__ == "__main__":
    from voi.parse_ivtff import parse_ivtff, add_n_token_columns
    ld, _ = parse_ivtff(str(common.data_path("voynich")))
    ld = add_n_token_columns(ld)
    run(ld)
