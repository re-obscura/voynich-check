"""
Фокус-проверка находки B: почему MI Войнича не затухает с дистанцией?
Возможные объяснения:
  (1) Тематическая связность (как в языке) — настоящий сигнал.
  (2) Артефакт: «UNK bucket». Топ-100 частых vs <UNK>; если редкие слова
      кластеризуются на соседних фолио (а фолио тематически однородны),
      MI(UNK,UNK) даёт плоский хвост — это не «язык», а структура документа.
  (3) Артефакт раздела: bio/recipes монотонны — длинные блоки одних слов.

Разделяем: MI(top,top) vs MI(top,UNK) vs MI(UNK,UNK). Если хвост держится
на UNK-UNK — это (2), структура документа, а не языка.
"""
import sys; sys.path.insert(0, ".")
import collections, math
import numpy as np
from voi.parse_ivtff import parse_ivtff, add_n_token_columns
from voi.common import data_path
from voynich_lm.analyze.round4 import voynich_words, control_words, shuffle_words


def mi_decomposed(words, frequent, max_dist=32):
    """MI разбита по классам пар: TT (top-top), TU, UU."""
    idx = [w if w in frequent else "<UNK>" for w in words]
    classes = {"TT":[], "TU":[], "UU":[]}
    for d in range(1, max_dist+1):
        joint = collections.Counter()
        n = 0
        for i in range(len(idx)-d):
            a, b = idx[i], idx[i+d]
            joint[(a,b)] += 1; n += 1
        if n == 0: continue
        # суммарный MI по парам класса
        for cls in classes:
            tot_pxy = 0
            for (a,b), c in joint.items():
                aU = (a=="<UNK>"); bU = (b=="<UNK>")
                if cls=="TT" and not aU and not bU: tot_pxy += c
                elif cls=="UU" and aU and bU: tot_pxy += c
                elif cls=="TU" and (aU ^ bU): tot_pxy += c
            classes[cls].append(tot_pxy/n)
    return classes


def run():
    print("="*68)
    print("ФОКУС: декомпозиция MI Войнича (артефакт UNK vs настоящий сигнал?)")
    print("="*68)
    texts = {"voynich": voynich_words(), "kjv": control_words("kjv"),
             "shuffle": shuffle_words(voynich_words())}
    for name, w in texts.items():
        counts = collections.Counter(w)
        frequent = {x for x,_ in counts.most_common(100)}
        cls = mi_decomposed(w, frequent)
        print(f"\n  {name}: доля пар по классу на дистанции")
        print(f"    {'dist':>5s} {'TT(top-top)':>12s} {'TU':>10s} {'UU(unk-unk)':>12s}")
        for d in [0,1,4,9,19]:
            if d < len(cls["TT"]):
                print(f"    {d+1:5d} {cls['TT'][d]*100:11.2f}% {cls['TU'][d]*100:9.2f}% {cls['UU'][d]*100:11.2f}%")
    print()
    print("  Интерпретация: если UU(unk-unk) у Войнича держится высоко на")
    print("  больших d — это структура ДОКУМЕНТА (тематически однородные фолио),")
    print("  а не языка. Если TT(top-top) — настоящий контекстный сигнал.")


if __name__ == "__main__":
    run()
