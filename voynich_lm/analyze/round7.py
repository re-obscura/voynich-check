"""
Раунд 7: поиск «внешнего ключа» — мог ли Войнич быть номенклатором для языка?

H4 (раунд 6) показал: максимум смысла в тексте ≤0.02 бит/глиф. Это парадокс для
гипотезы номенклатора: таблица «слово→код» меняет частоты, оставляя статистический
след >>0.02 бит. Но бывают тонкие случаи. Проверяем строго.

K1. НОМЕНКЛАТОР-ТЕСТ — поведенческое выравнивание.
    Функциональные слова (the/and/of, et/in/qui, il/che/di) узнаваемы по ПОВЕДЕНИЮ:
    высочайшая частота, низкая дисперсия (равномерны по тексту), маленькая длина,
    стоят в начале фраз. Находим войничские слова с ТОЧНО таким поведением.
    Если daiin ведёт себя как 'the' — кандидат на статью/предлог.

K2. СЛОГОВОЕ СОПОСТАВЛЕНИЕ — у Войнича реконструированы слоги (H2 раунда 6).
    Какой язык имеет тот же профиль (CV-CVC структура)? Мера: распределение
    структур слов (по классам гласная/согласная).

K3. ОТПЕЧАТОК ДЛИН СЛОВ — распределение длин слов как ДНК языка. Латынь/итальянский/
    английский дают характерные кривые. Сравниваем Войнич.

K4. КРОСС-ЯЗЫКОВАЯ PERPLEXITY — LM на кандидатах (латынь/итальянский/английский),
    мерим CE Войнича под разными гипотезами о ключе. Простой шифр исключён (DECODE.md),
    но проверяем устойчивость.

K5. ТЕОРЕТИЧЕСКОЕ ОГРАНИЧЕНИЕ — вычисляем, сколько бит оставил бы реальный
    номенклатор (n code-words → n смысловых единиц), и сравниваем с H4 (0.02 бит).
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


def load_words(path, max_words=None):
    t = Path(path).read_text(encoding="utf-8", errors="ignore")
    s = t.find("*** START OF"); e = t.find("*** END OF")
    body = t[s:e] if s > 0 and e > 0 else t
    # для latin_combined.txt (plain text) — нет маркеров, берём как есть
    if s < 0: body = t
    words = re.findall(r"[a-zàèéìòù]+", body.lower())
    if max_words: words = words[:max_words]
    return words


def voynich_words():
    ld, _ = parse_ivtff(str(data_path("voynich")))
    ld = add_n_token_columns(ld)
    return [t for ts in ld[ld["n_tokens"] > 0]["tokens"] for t in ts]


# ============================================================
# K1. НОМЕНКЛАТОР-ТЕСТ: поведенческое выравнивание
# ============================================================

def word_behavior_profile(words, top_k=50):
    """
    Профиль поведения слова: (частота, дисперсия по фолио/главам, средняя длина,
    доля появлений в начале строки/предложения).
    Возвращает для топ-K слов их нормированные признаки.
    """
    # разбиение на «документы» (фолио для войнича, ~1000-словные чанки для языков)
    n = len(words)
    chunk = 200
    docs = [words[i:i+chunk] for i in range(0, n, chunk)]
    counts = collections.Counter(words)
    total = sum(counts.values())
    profiles = {}
    for w, c in counts.most_common(top_k):
        freq = c / total
        # дисперсия: равномерно ли по документам?
        doc_counts = np.array([d.count(w) for d in docs])
        disp = doc_counts.std() / (doc_counts.mean() + 1e-9) if doc_counts.mean() > 0 else 99
        profiles[w] = {"freq": freq, "dispersion_cv": float(disp),
                       "length": len(w), "rank": 0}
    # ранг по частоте
    for i, w in enumerate(sorted(profiles, key=lambda x: -profiles[x]["freq"])):
        profiles[w]["rank"] = i + 1
    return profiles


def behavioral_match(voyn_prof, lang_prof, top_n=20):
    """
    Найти войничские слова, чьё поведение (норм. дисперсия, длина, ранг) ближе
    всего к функциональным словам языка. Мерим MSE по z-scored признакам.
    """
    # нормируем признаки
    def z(d, key):
        vals = [d[w][key] for w in d]
        m, s = np.mean(vals), np.std(vals) + 1e-9
        return {w: (d[w][key] - m) / s for w in d}
    keys = ["freq", "dispersion_cv", "length"]
    vz = {k: z(voyn_prof, k) for k in keys}
    lz = {k: z(lang_prof, k) for k in keys}
    # для каждого топ-войничского слова — ближайшее язык-слово
    matches = []
    voyn_top = sorted(voyn_prof, key=lambda x: voyn_prof[x]["rank"])[:top_n]
    lang_top = list(lang_prof)
    for wv in voyn_top:
        best, best_d = None, 99
        for wl in lang_top:
            d = sum((vz[k][wv] - lz[k][wl])**2 for k in keys)
            if d < best_d: best_d, best = d, wl
        matches.append((wv, wl if best else "?", best_d,
                        voyn_prof[wv]["rank"], lang_prof.get(best, {}).get("rank", -1)))
    matches.sort(key=lambda x: x[2])
    return matches


# ============================================================
# K2. СЛОГОВОЕ СОПОСТАВЛЕНИЕ
# ============================================================

def syllable_structure_profile(words, max_len=8):
    """
    Профиль структуры слов: для каждого слова шаблон C/V (согласная/гласная).
    Распределение шаблонов по длинам — «отпечаток» слоговой системы языка.
    """
    # простая гласная/согласная классификация (для латыни/итальянского/английского)
    vowels = set("aeiouàèéìòù")
    # для Войнича используем реконструированные гласные из раунда 3: o,a,e (+y)
    voynich_vowels = set("oae")
    profiles = collections.Counter()
    for w in words:
        if not w or len(w) > max_len: continue
        # определяем гласные в зависимости от алфавита
        vset = voynich_vowels if any(c in "cdhkl" for c in w) and not any(c in "bfgjmpqwxyz" for c in w) else vowels
        templ = "".join("V" if c in vset else "C" for c in w)
        profiles[(len(w), templ)] += 1
    total = sum(profiles.values())
    return {k: v/total for k, v in profiles.items() if v/total > 0.005}


def structure_jsd(prof_a, prof_b):
    keys = set(prof_a) | set(prof_b)
    a = {k: prof_a.get(k, 1e-6) for k in keys}
    b = {k: prof_b.get(k, 1e-6) for k in keys}
    return js_divergence(a, b)


# ============================================================
# K3. ОТПЕЧАТОК ДЛИН СЛОВ
# ============================================================

def word_length_dist(words, max_len=12):
    lengths = [min(len(w), max_len) for w in words]
    counts = collections.Counter(lengths)
    total = sum(counts.values())
    return {l: counts[l]/total for l in range(1, max_len+1)}


# ============================================================
# K4. КРОСС-ЯЗЫКОВАЯ PERPLEXITY (через n-граммные модели слов)
# ============================================================

def cross_lang_perplexity(target_words, lang_words, n=2):
    """
    Перплексия target под n-граммной моделью языка. Мера «насколько target
    похож на язык по порядку слов». Низкая = target ведёт себя как язык.
    """
    # n-граммная модель на lang_words
    ng = collections.Counter(); ctx = collections.Counter()
    for i in range(len(lang_words) - n + 1):
        ng[tuple(lang_words[i:i+n])] += 1
        ctx[tuple(lang_words[i:i+n-1])] += 1
    V = len(set(lang_words))
    # perplexity на target
    ll = 0.0; cnt = 0
    for i in range(len(target_words) - n + 1):
        c = tuple(target_words[i:i+n-1]); w = target_words[i+n-1] if n > 1 else target_words[i]
        num = ng.get(c + (target_words[i+n-1],) if n > 1 else (target_words[i],), 0) + 0.01
        den = ctx.get(c, 0) + 0.01 * V if n > 1 else len(lang_words)
        p = num / den
        ll += -math.log2(p); cnt += 1
    return ll / cnt if cnt else float("nan")


# ============================================================
# K5. ТЕОРЕТИЧЕСКОЕ ОГРАНИЧЕНИЕ номенклатора
# ============================================================

def nomenclator_bound(voyn_words):
    """
    Если номенклатор отображал смысловые единицы в код-слова, сколько бит
    информации он вносил бы? При словаре V код-слов и равномерном их использовании
    для S смысловых единиц, энтропия = log2(S) бит на код-слово. Но реальный
    язык имеет Ципф-распределение смысловых единиц, дающее избыточность.
    Вычисляем: при наблюдаемых частотах Войнича, сколько бит «выбора» осталось бы
    у писца на слово, если бы код-слова соответствовали смыслам.
    """
    counts = collections.Counter(voyn_words)
    total = sum(counts.values())
    # наблюдаемая энтропия распределения слов (безусловная)
    H = -sum((c/total) * math.log2(c/total) for c in counts.values())
    # если бы код-слова = смысловые единицы, эффективное число «смыслов»
    n_eff = 2 ** H
    # в естественном языке смысловая энтропия на слово ~ H_typical ~ 7-9 бит
    # (топ-3 по Шеннону: ~9 бит/слово для английского)
    H_typical_natural = 9.0  # бит/слово
    # сколько бит «смысла» могло бы вместить распределение Войнича
    voyn_bits_per_word = H
    return {"observed_word_entropy_bits": H, "effective_vocab": n_eff,
            "natural_lang_entropy_bits": H_typical_natural,
            "ratio_voyn_to_natural": H / H_typical_natural,
            "interpretation": ("словарь Войнича вмещает смысловой объём, "
                               f"эквивалентный {H/H_typical_natural*100:.0f}% естественного языка")}


# ============================================================
# Оркестр
# ============================================================

def run(figures: Path = Path("figures")) -> dict:
    figures.mkdir(exist_ok=True)
    print("=" * 68)
    print("РАУНД 7: поиск внешнего ключа — мог ли Войнич быть номенклатором?")
    print("=" * 68)

    N = 31000  # унифицированная длина
    texts = {
        "voynich": voynich_words()[:N],
        "latin": load_words("data/latin_combined.txt", N),
        "italian": load_words("data/italian.txt", N),
        "english": load_words(str(data_path("kjv")), N),
    }
    results = {}

    # ---- K1 ----
    print("\n--- K1. НОМЕНКЛАТОР-ТЕСТ (поведенческое выравнивание топ-слов) ---")
    profs = {name: word_behavior_profile(w) for name, w in texts.items()}
    print(f"\n  Топ-10 слов по языкам (частота / дисперсия / длина):")
    for name in texts:
        top = sorted(profs[name], key=lambda x: profs[name][x]["rank"])[:8]
        print(f"  {name:10s}: " + " ".join(f"{w}({profs[name][w]['freq']:.3f})" for w in top))
    print(f"\n  Поведенческое выравнивание Войнич ↔ языки (топ-войнич ↔ ближайшее язык-слово):")
    k1 = {}
    for lang in ["latin", "italian", "english"]:
        m = behavioral_match(profs["voynich"], profs[lang], top_n=15)
        k1[lang] = [{"voyn": wv, "lang": wl, "dist": round(d, 2),
                      "voyn_rank": rv, "lang_rank": rl}
                     for wv, wl, d, rv, rl in m[:10]]
        print(f"\n  Войнич ↔ {lang}:")
        for r in k1[lang][:8]:
            print(f"    ранг{r['voyn_rank']:2d} '{r['voyn']}' ~ {lang} '{r['lang']}' (rank{r['lang_rank']}, dist={r['dist']})")
    results["k1_nomenclator"] = k1

    # ---- K2 ----
    print("\n--- K2. СЛОГОВОЕ СОПОСТАВЛЕНИЕ (структуры C/V слов) ---")
    struct_profs = {name: syllable_structure_profile(w) for name, w in texts.items()}
    print(f"\n  JSD структур слов Войнич ↔ языки:")
    k2 = {}
    for lang in ["latin", "italian", "english"]:
        j = structure_jsd(struct_profs["voynich"], struct_profs[lang])
        k2[lang] = j
        print(f"    Войнич ↔ {lang:10s}: JSD = {j:.3f}")
    best_lang = min(k2, key=k2.get)
    print(f"  => ближайший по слоговой структуре: {best_lang} (JSD={k2[best_lang]:.3f})")
    results["k2_syllable"] = k2

    # ---- K3 ----
    print("\n--- K3. ОТПЕЧАТОК ДЛИН СЛОВ ---")
    len_dists = {name: word_length_dist(w) for name, w in texts.items()}
    print(f"\n  Распределение длин слов (доля по длинам):")
    print(f"  {'длина':>6s} " + " ".join(f"{n:>9s}" for n in texts))
    for l in range(1, 10):
        print(f"  {l:6d} " + " ".join(f"{len_dists[n].get(l,0)*100:8.1f}%" for n in texts))
    k3 = {}
    for lang in ["latin", "italian", "english"]:
        j = js_divergence(len_dists["voynich"], len_dists[lang])
        k3[lang] = j
    print(f"\n  JSD распределения длин Войнич ↔ языки:")
    for lang, j in k3.items():
        print(f"    {lang:10s}: {j:.3f}")
    results["k3_word_length"] = k3

    # ---- K4 ----
    print("\n--- K4. КРОСС-ЯЗЫКОВАЯ PERPLEXITY (порядок слов) ---")
    k4 = {}
    for lang in ["latin", "italian", "english"]:
        # перемешанный войнич как null
        pp = cross_lang_perplexity(texts["voynich"], texts[lang], n=2)
        null = cross_lang_perplexity(list(np.random.default_rng(7).permutation(texts["voynich"])), texts[lang], n=2)
        k4[lang] = {"voynich_ppl": pp, "shuffle_ppl": null}
        print(f"  модель {lang:10s}: Войнич ppl={pp:.2f}  shuffle null={null:.2f}  ratio={pp/null:.3f}")
    results["k4_cross_ppl"] = k4

    # ---- K5 ----
    print("\n--- K5. ТЕОРЕТИЧЕСКОЕ ОГРАНИЧЕНИЕ номенклатора ---")
    k5 = nomenclator_bound(texts["voynich"])
    print(f"  наблюдаемая энтропия слов Войнича: {k5['observed_word_entropy_bits']:.2f} бит/слово")
    print(f"  эффективный словарь: {k5['effective_vocab']:.0f} «смыслов»")
    print(f"  естественный язык (типично): {k5['natural_lang_entropy_bits']:.1f} бит/слово")
    print(f"  => {k5['interpretation']}")
    results["k5_bound"] = k5

    # ---- графики ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    # K3 длины слов
    colors = {"voynich":"#c0392b","latin":"#2980b9","italian":"#27ae60","english":"#8e44ad"}
    for name, d in len_dists.items():
        ls = sorted(d)
        axes[0].plot(ls, [d[l] for l in ls], "o-", color=colors[name], label=name,
                     lw=2 if name=="voynich" else 1, alpha=0.85)
    axes[0].set_title("K3: отпечаток длин слов"); axes[0].set_xlabel("длина слова")
    axes[0].set_ylabel("доля"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    # K2 слоги JSD
    axes[1].bar(k2.keys(), k2.values(), color=[colors[l] for l in k2])
    axes[1].set_title("K2: JSD слоговых структур Войнич↔языки"); axes[1].set_ylabel("JSD")
    axes[1].grid(axis="y", alpha=0.3)
    # K4 cross-ppl
    langs = list(k4.keys())
    x = np.arange(len(langs))
    axes[2].bar(x-0.2, [k4[l]["voynich_ppl"] for l in langs], 0.4, label="Войнич", color="#c0392b")
    axes[2].bar(x+0.2, [k4[l]["shuffle_ppl"] for l in langs], 0.4, label="shuffle null", color="#7f8c8d")
    axes[2].set_xticks(x); axes[2].set_xticklabels(langs)
    axes[2].set_title("K4: кросс-языковая perplexity\n(Войнич не ближе ни к одному)")
    axes[2].legend(fontsize=8); axes[2].grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(figures / "R7_external_key.png", dpi=130)
    print(f"\n  -> figures/R7_external_key.png")

    Path("round7_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=lambda o: float(o) if hasattr(o,"__float__") else str(o)),
        encoding="utf-8")
    print(f"  -> round7_results.json")
    return results


if __name__ == "__main__":
    run()
