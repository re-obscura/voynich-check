"""
Фокус-проверка: работает ли section-проба (91%) ТОЛЬКО через Currier?
Если да, то внутри одного диалекта точность упадёт к baseline.
Если нет — section несёт независимый сигнал.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
import torch
from voynich_lm.data import build_voynich_dataset
from voynich_lm.train import load_model
from voynich_lm.analyze.probing import folio_hidden_vectors, probe_labels, run_probe

res = {}
ds = build_voynich_dataset()
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = load_model("voynich", device=device)
X, folios = folio_hidden_vectors(ds, model, device)
sec, cur, scr = probe_labels(ds, folios)

print("=" * 68)
print("ФОКУС: section-проба внутри одного диалекта (контроль Currier)")
print("=" * 68)
print()

# только Currier B (72 фолио, несколько секций внутри)
for dialect in ("A", "B"):
    mask = (cur == dialect)
    if mask.sum() < 20:
        continue
    Xd, secd = X[mask], sec[mask]
    print(f"  Внутри Currier {dialect} (n={mask.sum()} фолио):")
    # сколько секций представлено
    import collections
    csec = collections.Counter(secd.tolist() if hasattr(secd,'tolist') else secd)
    print(f"    секции: {dict(csec)}")
    r = run_probe(Xd, secd, f"section_in_{dialect}", min_class_size=3)
    print(f"    section-проба: acc={r['acc']*100:.1f}% baseline={r['baseline']*100:.1f}% "
          f"(n={r['n']}, классов={r['n_classes']})")
    if r["acc"] - r["baseline"] < 0.1:
        print(f"    => точность УПАЛА к baseline: section-сигнал идёт ЧЕРЕЗ Currier")
    else:
        print(f"    => точность ВЫШЕ baseline: section несёт НЕЗАВИСИМЫЙ сигнал (не только Currier)")
    print()

# scribe внутри B
maskB = (cur == "B")
if maskB.sum() >= 20:
    Xd, scrd = X[maskB], scr[maskB]
    r = run_probe(Xd, scrd, "scribe_in_B", min_class_size=3)
    print(f"  scribe-проба внутри Currier B: acc={r['acc']*100:.1f}% baseline={r['baseline']*100:.1f}%")
    print(f"    => {'независимый' if r['acc']-r['baseline']>0.1 else 'через Currier/section'}")
