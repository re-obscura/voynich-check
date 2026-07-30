"""
Фокус-Crosscheck: систематическая разница surprisal A (~1.96) vs B (~1.77).
Это артефакт обучения (модель видит оба, но B «проще») или свойство B?
Проверяем: обучим отдельные 3-граммы на A-only и B-only, сравним H.
Если B «механичнее», то его собственная H будет НИЖЕ, чем у A — независимо
от смешанной модели. Это подтверждает тезис «B — более шаблонный генератор».
"""
import sys; sys.path.insert(0, ".")
import math, collections
import numpy as np
from voi.parse_ivtff import parse_ivtff, add_n_token_columns
from voi.common import data_path
from voynich_lm.analyze.round5 import NGramGlyphModel

def stream_for(lines_df, currier=None):
    P = lines_df[lines_df["n_tokens"] > 0]
    if currier:
        P = P[P["currier"] == currier]
    s = ""
    for toks in P["tokens"]:
        s += ".".join(toks) + "."
    return s

def ce_of_model_on_stream(model, stream):
    order = model.order
    ce = 0.0; n = 0
    for i in range(order, len(stream)):
        ctx = stream[i-order:i]
        cands = model.context_counts.get(ctx)
        if not cands: continue
        total = sum(cands.values())
        p = (cands.get(stream[i],0)+0.01)/(total+0.01*len(model.vocab))
        ce += -math.log2(p); n += 1
    return ce/n

ld, _ = parse_ivtff(str(data_path("voynich")))
ld = add_n_token_columns(ld)

print("="*68)
print("ФОКУС: Currier A vs B — кто «механичнее»? (раздельные 3-граммы)")
print("="*68)
sA = stream_for(ld, "A"); sB = stream_for(ld, "B")
print(f"  поток A: {len(sA)} глифов, B: {len(sB)} глифов")
mA = NGramGlyphModel(order=3); mA.fit(sA)
mB = NGramGlyphModel(order=3); mB.fit(sB)
# собственная CE (на своём тексте)
ceA = ce_of_model_on_stream(mA, sA)
ceB = ce_of_model_on_stream(mB, sB)
# кросс: модель A на потоке B и наоборот
ceAB = ce_of_model_on_stream(mA, sB)
ceBA = ce_of_model_on_stream(mB, sA)
print(f"\n  собственная CE:  A={ceA:.3f}  B={ceB:.3f} бит/глиф")
print(f"  кросс CE:        A→B={ceAB:.3f}  B→A={ceBA:.3f}")
print(f"\n  => {'B МЕХАНИЧНЕЕ (ниже собственная CE)' if ceB < ceA else 'A механичнее'}")
print(f"  => {'A и B — РАЗНЫЕ машины' if (ceAB-ceB>0.05 and ceBA-ceA>0.05) else 'A и B похожи'}")
print(f"     кросс-A→B vs self-B: {ceAB-ceB:+.3f} (насколько B чужд модели A)")
print(f"     кросс-B→A vs self-A: {ceBA-ceA:+.3f} (насколько A чужд модели B)")
