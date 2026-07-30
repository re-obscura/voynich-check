"""
B: генеративный инвертирующий тест.

Модель, обученная на Войниче, генерирует синтетические фолио. Затем по ним
прогоняется СУЩЕСТВУЮЩАЯ батарея структурных гипотез (переиспользуем функции
из voi/): line-edge JSD + суффикс -am (H1), повторы n-грамм (H2), self-citation
(H5), H_{2|1} (entropy). Если генеративная модель БЕЗ СМЫСЛА воспроизводит
статистический отпечаток Войнича -> подтверждение тезиса «имитация без смысла».

Структура синтетики: создаём lines_df того же формата, что parse_ivtff
(столбцы tokens, locus_type, folio, section, quire, para_pos, ...), чтобы
функции voi/ отработали без изменений.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from voynich_lm.train import load_model


# ============================================================
#  Генерация синтетических фолио
# ============================================================

SEP_GLYPH = "."
EOS = "<eos>"


@torch.no_grad()
def generate_folio(model, tok, n_chars: int = 2400, temperature: float = 1.0,
                    seed: int = 0, device: str = "cpu") -> str:
    """
    Генерация одного «фолио» (потока глифов). Затравка = <bos>.
    Возвращает декодированную строку (с SEP_GLYPH и спецтокенами как маркерами).
    """
    model.eval()
    torch.manual_seed(seed)
    bos = tok.bos_id
    eos = tok.eos_id
    ids = [bos]
    for _ in range(n_chars):
        x = torch.tensor([ids[-model.cfg["max_ctx"]:]], device=device, dtype=torch.long)
        logits = model(x)[0][0, -1] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1).item()
        if nxt == eos:
            # закончить "строку"/абзац — продолжим генерацию, eos как разделитель
            ids.append(eos)
        else:
            ids.append(nxt)
        if len(ids) > n_chars:
            break
    return tok.decode(ids)


def synthetic_to_lines_df(generated: list[str], tok,
                           folio_prefix: str = "synt") -> pd.DataFrame:
    """
    Преобразовать сгенерированные потоки в lines_df того же формата, что parse_ivtff.
    generated: список строк (по одной на синтетическое фолио).

    Логика восстановления структуры:
      - спецтокен <eos> разделяет «строки» (по нашей схеме данных eos = конец
        строки ИЛИ смена абзаца; здесь трактуем как граница строки/абзаца).
      - внутри строки SEP_GLYPH ('.') разделяет «слова».
    """
    eos_s = EOS
    sep_s = SEP_GLYPH
    rows = []
    for fi, gen in enumerate(generated):
        folio = f"{folio_prefix}{fi:03d}"
        # разобьём на «строки» по eos; пустые пропустим
        lines = [l for l in gen.split(eos_s) if l.strip(sep_s)]
        for li, line in enumerate(lines):
            # слова = сегменты, разделённые SEP_GLYPH, непустые
            words = [w for w in line.split(sep_s) if w and w != "<bos>"]
            if not words:
                continue
            rows.append({
                "folio": folio,
                "folio_num": fi,
                "locus_seq": li,
                "locus_type": "P0",
                "section": "synthetic",
                "quire": folio,            # фиктивная «тетрадь» = фолио
                "currier": "S",
                "scribe": "0",
                "para_pos": (0 if li == 0 else (2 if li == len(lines)-1 else 2)),
                "para_start": (li == 0),
                "tokens": words,
                "raw_text": sep_s.join(words),
                "n_tokens": len(words),
                "tokens_str": " ".join(words),
                "page_line_index": li,
                "page_n_lines": len(lines),
                "top_half": li < len(lines)/2,
            })
    return pd.DataFrame(rows)


# ============================================================
#  Батарея структурных тестов на синтетике
# ============================================================

def run_structural_battery(lines_df: pd.DataFrame) -> dict:
    """Прогон H1(line-edge)+H2(n-grams)+H5(self-cite)+entropy на lines_df."""
    from voi import h1_position, h2_ngrams, h5_selfcite, entropy
    out = {}

    # --- H1: line-edge JSD + суффикс -am ---
    P = lines_df[lines_df["locus_type"].str.startswith("P") &
                 (lines_df["tokens"].apply(len) > 0)].copy()
    try:
        jsd_f, jsd_l, nf, nl, nm = h1_position.line_edge_jsd(P)
        am = h1_position.suffix_am_line_edge(P)
        out["h1"] = {"jsd_first_vs_mid": jsd_f, "jsd_last_vs_mid": jsd_l,
                     "am_last_pct": am["am_last"]*100, "am_mid_pct": am["am_mid"]*100,
                     "am_ratio": am["ratio_last_mid"]}
    except Exception as e:
        out["h1"] = {"error": str(e)}

    # --- H2: повторы n-грамм (тетради фиктивны, мера — доля повторов 4-грамм) ---
    try:
        # все «тетради» = фолио; считаем сколько 4-грамм повторяются на >=3 фолио
        import collections
        from voi import common
        ng_folios = collections.defaultdict(set)
        for toks, folio in zip(P["tokens"], P["folio"]):
            for ng in common.ngrams(list(toks), 4):
                ng_folios[ng].add(folio)
        n_repeated = sum(1 for fs in ng_folios.values() if len(fs) >= 3)
        max_folios = max((len(fs) for fs in ng_folios.values()), default=0)
        # соседние дубли биграмм
        n_total = n_dup = 0
        for toks in P["tokens"]:
            for a, b in common.ngrams(list(toks), 2):
                n_total += 1
                if a == b:
                    n_dup += 1
        out["h2"] = {"n_unique_4grams": len(ng_folios),
                     "n_4grams_on_ge3_folios": n_repeated,
                     "max_folios_one_4gram": max_folios,
                     "adj_dup_bigram_pct": (n_dup/n_total*100) if n_total else 0.0}
    except Exception as e:
        out["h2"] = {"error": str(e)}

    # --- H5: self-citation ---
    try:
        # подмножество run, нужное для числа: scr при N=1,2
        Pc = P.sort_values(["folio_num", "locus_seq"]).reset_index(drop=True)
        lines_tokens = [list(t) for t in Pc["tokens"]]
        scr1 = h5_selfcite.self_citation_rate(lines_tokens, window=1)
        scr2 = h5_selfcite.self_citation_rate(lines_tokens, window=2)
        out["h5"] = {"scr_N1": scr1, "scr_N2": scr2}
    except Exception as e:
        out["h5"] = {"error": str(e)}

    # --- entropy: H_{2|1} ---
    try:
        toks = [t for ts in P["tokens"] for t in ts]
        h21 = entropy.conditional_entropy_chars(toks, sep=" ")
        out["entropy"] = {"H_2_1_bits": h21}
    except Exception as e:
        out["entropy"] = {"error": str(e)}

    return out


# ============================================================
#  Оркестр: синтетика Войнича vs реальный Войнич vs синтетика контроля
# ============================================================

def run(n_synthetic_folios: int = 20, chars_per_folio: int = 2400,
        temperature: float = 0.9, device: str | None = None,
        figures: Path = Path("figures"), out: dict | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    figures.mkdir(exist_ok=True)
    from voynich_lm.data import build_voynich_dataset

    ds = build_voynich_dataset()
    # реальный Войнич (только train-фолио, чтобы не сравнивать с тестом)
    ld_real = _real_train_lines_df(ds)

    # --- синтетика модели voynich ---
    print("=== B: генеративный тест ===")
    print(f"  генерация {n_synthetic_folios} синтетических фолио (T={temperature})...")
    model, tok = load_model("voynich", device=device)
    gens = [generate_folio(model, tok, n_chars=chars_per_folio,
                            temperature=temperature, seed=100+i, device=device)
            for i in range(n_synthetic_folios)]
    synt_df = synthetic_to_lines_df(gens, tok)

    battery_real = run_structural_battery(ld_real)
    battery_synt = run_structural_battery(synt_df)

    # референс: синтетика kjv-модели (если обучена)
    battery_kjv_synt = None
    try:
        model_k, tok_k = load_model("kjv", device=device)
        # kjv-модель использует латинский алфавит; токенизатор другой.
        gens_k = [generate_folio(model_k, tok_k, n_chars=chars_per_folio,
                                  temperature=temperature, seed=200+i, device=device)
                  for i in range(n_synthetic_folios)]
        synt_k_df = synthetic_to_lines_df(gens_k, tok_k)
        battery_kjv_synt = run_structural_battery(synt_k_df)
    except FileNotFoundError:
        print("  kjv-модель не обучена — референс синтетики пропущен")

    results = {"real_voynich": battery_real, "synthetic_voynich": battery_synt,
               "synthetic_kjv": battery_kjv_synt}

    _print_comparison(results)
    _plot_comparison(results, figures)

    if out is not None:
        out["generative"] = results
    return results


def _real_train_lines_df(ds) -> pd.DataFrame:
    """Собрать lines_df реального Войнича только из train-фолио (для fair compare)."""
    from voi.common import data_path
    from voi.parse_ivtff import parse_ivtff, add_n_token_columns
    ld, _ = parse_ivtff(str(data_path("voynich")))
    ld = add_n_token_columns(ld)
    ld = ld[ld["n_tokens"] > 0]
    ld = ld[ld["folio"].isin(ds.train_folios)].copy()
    return ld


def _print_comparison(results):
    print("\n  --- Сравнение: real Voynich | synthetic-Voynich | synthetic-KJV ---")
    rv, sv = results["real_voynich"], results["synthetic_voynich"]
    sk = results.get("synthetic_kjv") or {}
    def g(d, *ks, default="—"):
        cur = d
        for k in ks:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur
    print(f"  {'metric':32s} {'real':>10s} {'syn-V':>10s} {'syn-K':>10s}")
    def row(name, path, fmt="{:.3f}"):
        vals = [g(rv, *path), g(sv, *path), g(sk, *path)]
        cells = []
        for v in vals:
            try:
                cells.append(fmt.format(float(v)))
            except (ValueError, TypeError):
                cells.append(str(v))
        print(f"  {name:32s} {cells[0]:>10s} {cells[1]:>10s} {cells[2]:>10s}")
    row("H1 jsd_first_vs_mid", ("h1", "jsd_first_vs_mid"))
    row("H1 jsd_last_vs_mid", ("h1", "jsd_last_vs_mid"))
    row("H1 -am last %", ("h1", "am_last_pct"), "{:.2f}")
    row("H1 -am ratio (last/mid)", ("h1", "am_ratio"), "{:.1f}")
    row("H2 n_4grams on >=3 folios", ("h2", "n_4grams_on_ge3_folios"), "{:.0f}")
    row("H2 adj dup bigram %", ("h2", "adj_dup_bigram_pct"), "{:.2f}")
    row("H5 self-cite N=1", ("h5", "scr_N1"), "{:.4f}")
    row("entropy H_2|1 (bit)", ("entropy", "H_2_1_bits"), "{:.3f}")


def _plot_comparison(results, figures):
    rv, sv = results["real_voynich"], results["synthetic_voynich"]
    metrics = [
        ("H1 jsd(first,mid)", ("h1", "jsd_first_vs_mid"), "{:.2f}"),
        ("H1 -am ratio", ("h1", "am_ratio"), "{:.1f}"),
        ("H2 adj-dup %", ("h2", "adj_dup_bigram_pct"), "{:.2f}"),
        ("H5 self-cite N1", ("h5", "scr_N1"), "{:.3f}"),
        ("entropy H2|1", ("entropy", "H_2_1_bits"), "{:.2f}"),
    ]
    labels = [m[0] for m in metrics]
    def getval(d, path):
        try:
            return float(d[path[0]][path[1]])
        except Exception:
            return float("nan")
    real_v = [getval(rv, m[1]) for m in metrics]
    synt_v = [getval(sv, m[1]) for m in metrics]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.2, real_v, 0.4, label="real Voynich", color="#c0392b")
    ax.bar(x + 0.2, synt_v, 0.4, label="synthetic (LM)", color="#2980b9")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("Генеративный тест: отпечаток реального Войнича vs синтетики LM")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "B_generative.png", dpi=130)
    print(f"  -> {figures/'B_generative.png'}")


if __name__ == "__main__":
    res = run()
