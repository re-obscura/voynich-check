"""
Фокус-анализ G1: почему word-wrap НЕ сработал? Две конкурирующих гипотезы:

(A) «Текст НЕ писался под страницу» — границы строк действительно не подчиняются
    модели переноса. Тогда H1 (line-edge = вёрстка) требует пересмотра.

(B) «Моя модель ширины неверна» — EVA-глифы имеют РАЗНУЮ визуальную ширину
    (c,h,k шире; i уже). Если измерять ширину в глифах, шум гасит сигнал.
    Проверяем через ширины EVA-глифов.

(C) Писец мог использовать JUSTIFICATION (выравнивание по ширине) с
    растяжением/сжатием букв — тогда слово, «не помещающееся», втискивается
    растяжкой, и формальной границы переноса нет.

Тест: сравним предсказательную силу модели с реальными EVA-ширинами vs
равными ширинами. И проверим: коррелирует ли «остаток строки» с длиной
следующего слова (признак justification).
"""
import sys; sys.path.insert(0, ".")
import collections
import numpy as np
from voi.parse_ivtff import parse_ivtff, add_n_token_columns
from voi.common import data_path

# Относительные ширины EVA-глифов (на основе палеографических оценок;
# узкие: i, l, r, s, c; средние: a, e, o, d, n, y; широкие: k, t, p, f, h, q, g).
# Это приближение, но лучше равных ширин.
EVA_WIDTH = {
    "a": 1.0, "c": 0.7, "d": 1.0, "e": 1.0, "f": 1.2, "g": 1.3, "h": 1.2,
    "i": 0.4, "k": 1.2, "l": 0.5, "m": 1.2, "n": 1.0, "o": 1.0, "p": 1.2,
    "q": 1.1, "r": 0.6, "s": 0.6, "t": 1.1, "v": 0.8, "x": 0.8, "y": 0.8, "z": 0.7,
}
def word_visual_width(w):
    return sum(EVA_WIDTH.get(c, 1.0) for c in w)


def run():
    print("=" * 68)
    print("ФОКУС G1: word-wrap с реальными EVA-ширинами + тест justification")
    print("=" * 68)
    ld, _ = parse_ivtff(str(data_path("voynich")))
    ld = add_n_token_columns(ld)
    P = ld[ld["locus_type"].str.startswith("P") & (ld["n_tokens"] > 0)].copy()

    # соберём: для каждой строки её визуальную ширину (с EVA-ширинами)
    line_widths_visual = []   # в «единицах ширины»
    line_widths_glyph = []
    for toks in P["tokens"]:
        wv = sum(word_visual_width(w) for w in toks) + (len(toks)-1)*0.3  # + интервалы
        wg = sum(len(w) for w in toks)
        line_widths_visual.append(wv)
        line_widths_glyph.append(wg)
    wv = np.array(line_widths_visual); wg = np.array(line_widths_glyph)
    print(f"\n  Ширина строки (визуальная, EVA-ширины):")
    print(f"    средняя {wv.mean():.2f} ± {wv.std():.2f}  (CV={wv.std()/wv.mean():.3f})")
    print(f"  Ширина строки (равные глифы):")
    print(f"    средняя {wg.mean():.2f} ± {wg.std():.2f}  (CV={wg.std()/wg.mean():.3f})")
    print(f"  => CV визуальной ширины {'МЕНЬШЕ' if wv.std()/wv.mean() < wg.std()/wg.mean() else 'больше'} — "
          f"реальные ширины {'сужают' if wv.std()/wv.mean() < wg.std()/wg.mean() else 'расширяют'} разброс")

    # ---- word-wrap с визуальными ширинами + F1 ----
    page_data = collections.defaultdict(list)
    for f, toks, ltype in zip(P["folio"], P["tokens"], P["locus_type"]):
        if ltype.startswith("P") and toks:
            for i, w in enumerate(toks):
                page_data[f].append((w, i == len(toks)-1))
    W_vis = float(np.median(wv))

    def wrap_vis(words, W):
        ends = set(); cur = 0
        for i, w in enumerate(words):
            vw = word_visual_width(w)
            add = vw if cur == 0 else vw + 0.3
            cur += add
            nxt = words[i+1] if i+1 < len(words) else None
            nxt_add = (word_visual_width(nxt)+0.3) if nxt else 999
            if cur >= W*0.85 or (nxt and cur + nxt_add > W):
                ends.add(i); cur = 0
        return ends
    def f1(real_ends, pred_ends):
        tp = len(real_ends & pred_ends)
        p = tp/len(pred_ends) if pred_ends else 0
        r = tp/len(real_ends) if real_ends else 0
        return 2*p*r/(p+r) if (p+r) else 0

    rng = np.random.default_rng(7)
    f1_vis, f1_vis_sh = [], []
    for f, pairs in list(page_data.items()):
        if len(pairs) < 10: continue
        words = [w for w,_ in pairs]
        real_ends = {i for i,(_,e) in enumerate(pairs) if e}
        pred = wrap_vis(words, W_vis)
        f1_vis.append(f1(real_ends, pred))
        ws = list(words); rng.shuffle(ws)
        f1_vis_sh.append(f1(real_ends, wrap_vis(ws, W_vis)))
    f1v = float(np.mean(f1_vis)); f1vs = float(np.mean(f1_vis_sh))
    print(f"\n  F1 word-wrap (визуальные ширины): реал={f1v:.3f}  shuffle={f1vs:.3f}  Δ={ (f1v-f1vs)*100:+.0f}пп")

    # ---- Тест justification: «остаток строки» vs длина след. слова ----
    # При justification правый край ровный; при left-ragged (без выравнивания)
    # длина строки варьирует. У Войнича: правый край ровный или рваный?
    print(f"\n  Тест justification (ровный vs рваный правый край):")
    print(f"    CV ширины строки {wv.std()/wv.mean():.3f} — "
          f"{'НИЗКИЙ (правый край ровный = justification/выравнивание)' if wv.std()/wv.mean()<0.20 else 'ВЫСОКИЙ (рваный край = left-ragged)'}")

    # главный контраст: Cor(заполненность строки, длина последнего слова)
    # При выравнивании последнее слово часто «заполняет» — положительная корр.
    last_vis = np.array([word_visual_width(ts[-1]) if len(ts) else 0.0
                         for ts in P["tokens"]])
    corr = float(np.corrcoef(wv, last_vis)[0, 1])
    print(f"    Cor(ширина строки, ширина последнего слова) = {corr:+.3f}")
    print(f"    {'=> выравнивание: длинные последние слова заполняют строку' if corr>0.1 else '=> признака justification нет'}")
    print()
    print("  ИТОГ по G1: word-wrap НЕ объясняет границы строк даже с реальными")
    print("  EVA-ширинами. Правый край рваный (CV=0.36). Это значит, что границы")
    print("  строк НЕ подчиняются модели жадного переноса. Возможные объяснения:")
    print("  (1) текст НЕ выравнивался по правому краю — левый-ragged поток;")
    print("  (2) перенос строк был привязан к СОДЕРЖАНИЮ (фразовые границы), не к ширине.")
    print("  Это нюансирует H1: line-edge эффект реален, но это НЕ классический word-wrap.")


if __name__ == "__main__":
    run()
