"""
E: position-conditioned perplexity.

Для каждого глифа в потоке — его нормализованная позиция в строке (по словам)
и позиция строки в абзаце. Группируем NLL модели по позиции -> непрерывный,
модельный взгляд на line-edge эффект (H1). Если perplexity растёт к правому
краю строки (где действует «заполнитель» -am), это генеративное подтверждение
line-edge-эффекта.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from voynich_lm.train import load_model


def run(ctx: int = 256, device: str | None = None,
        figures: Path = Path("figures"), out: dict | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    figures.mkdir(exist_ok=True)
    from voi.common import data_path
    from voi.parse_ivtff import parse_ivtff, add_n_token_columns
    from voynich_lm.data import build_voynich_dataset

    ds = build_voynich_dataset()
    model, tok = load_model("voynich", device=device)

    # соберём позиционно-размеченный поток: для каждого токена его
    # (word_index_in_line, line_n_words, line_index_in_para)
    ld, _ = parse_ivtff(str(data_path("voynich")))
    ld = add_n_token_columns(ld)
    P = ld[ld["locus_type"].str.startswith("P") & (ld["n_tokens"] > 0)].copy()

    # поток токенов с метапозицией. sep='.' — отдельный токен-разделитель.
    # Будем считать позицию в строке по словам; глифы «внутри слова» наследуют
    # позицию слова.
    # nll_by_pos[word_idx] = список NLL глифов этого слова
    nll_by_word_pos: dict[int, list[float]] = {}
    nll_by_norm_pos: list[tuple[float, float]] = []  # (norm_pos_in_line, nll)
    nll_by_line_pos: dict[int, list[float]] = {}     # para_pos: 0 first / 1 last / 2 mid

    # соберём весь поток и параллельно — массив позиции (word-индекс в строке)
    # по каждому глифу. Для оценки NLL прогоним модель окнами ctx.
    all_ids: list[int] = []
    glyph_pos: list[float] = []     # нормализованная позиция слова в строке [0..1]
    glyph_para: list[int] = []      # para_pos (0/1/2)
    sep_id = tok.stoi["."]
    for toks, ppos in zip(P["tokens"], P["para_pos"]):
        nwords = len(toks)
        for wi, w in enumerate(toks):
            norm = wi / max(nwords - 1, 1)  # 0..1
            for ch in w:
                i = tok.stoi.get(ch)
                if i is not None:
                    all_ids.append(i)
                    glyph_pos.append(norm)
                    glyph_para.append(int(ppos))
            all_ids.append(sep_id)
            glyph_pos.append(norm)   # разделитель наследует позицию слова
            glyph_para.append(int(ppos))

    data = np.array(all_ids, dtype=np.int64)
    glyph_pos = np.array(glyph_pos, dtype=np.float32)
    glyph_para = np.array(glyph_para, dtype=np.int64)

    # оценим NLL непересекающимися окнами
    n_windows = (len(data) - 1) // ctx
    all_nll = np.zeros(len(data), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for k in range(n_windows):
            i0 = k * ctx
            x = torch.tensor(data[i0:i0+ctx], device=device, dtype=torch.long)[None]
            tgt = torch.tensor(data[i0+1:i0+1+ctx], device=device, dtype=torch.long)[None]
            nll, _ = model.nll_per_position(x, tgt)  # (1,ctx) нит
            all_nll[i0+1:i0+1+ctx] = nll[0].cpu().numpy()
    nll_bits = all_nll / np.log(2)

    # --- агрегация по нормализованной позиции в строке ---
    # разобьём на 10 бинов [0,0.1),...[0.9,1.0]
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_nll = []
    for b in range(10):
        mask = (glyph_pos >= bins[b]) & (glyph_pos < bins[b+1])
        if b == 9:
            mask = (glyph_pos >= bins[b]) & (glyph_pos <= bins[b+1])
        vals = nll_bits[mask]
        bin_nll.append(float(np.mean(vals)) if len(vals) else float("nan"))

    # --- агрегация по para_pos (first/last/mid строка абзаца) ---
    para_nll = {}
    for pp, name in [(0, "first"), (1, "last"), (2, "mid")]:
        vals = nll_bits[glyph_para == pp]
        para_nll[name] = float(np.mean(vals)) if len(vals) else float("nan")

    print("=== E: position-conditioned perplexity ===")
    print("  NLL (бит/глиф) по норм. позиции слова в строке:")
    print("   " + "  ".join(f"{c:.1f}:{bin_nll[i]:.2f}" for i, c in enumerate(bin_centers)))
    print(f"  по позиции строки в абзаце: first={para_nll['first']:.3f} "
          f"mid={para_nll['mid']:.3f} last={para_nll['last']:.3f}")

    # --- график ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(bin_centers, bin_nll, "o-", color="#c0392b")
    ax1.set_xlabel("норм. позиция слова в строке (0=первое, 1=последнее)")
    ax1.set_ylabel("NLL (бит/глиф)")
    ax1.set_title("Перплексия по позиции в строке\n(line-edge эффект генеративно)")
    ax1.grid(alpha=0.3)
    names = ["first", "mid", "last"]
    ax2.bar(names, [para_nll[n] for n in names], color=["#2980b9", "#95a5a6", "#c0392b"])
    ax2.set_ylabel("NLL (бит/глиф)")
    ax2.set_title("Перплексия по позиции строки\nв абзаце (first/mid/last)")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures / "E_position.png", dpi=130)
    print(f"  -> {figures/'E_position.png'}")

    results = {"nll_by_line_pos_bins": dict(zip([f"{c:.1f}" for c in bin_centers], bin_nll)),
               "nll_by_para_pos": para_nll}
    if out is not None:
        out["position"] = results
    return results


if __name__ == "__main__":
    run()
