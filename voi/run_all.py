"""
Сводный прогон всех гипотез + сохранение результатов в results.json.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

from voi import common
from voi.parse_ivtff import parse_ivtff, add_n_token_columns
from voi import entropy, h1_position, h2_ngrams, h3_cluster, h4_syntax, h5_selfcite


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(common.data_path("voynich"))
    print("# Загрузка корпуса:", path)
    ld, folios = parse_ivtff(path)
    ld = add_n_token_columns(ld)
    txt = ld[ld["n_tokens"] > 0]
    print(f"  фолио: {len(folios)}, строк с текстом: {len(txt)}")
    print()

    results = {}
    print(">>> Матчасть / энтропия")
    results["entropy"] = entropy.run(txt)
    print()
    print(">>> Гипотеза 1")
    results["h1"] = h1_position.run(ld)
    print()
    print(">>> Гипотеза 2")
    results["h2"] = h2_ngrams.run(ld)
    print()
    print(">>> Гипотеза 3")
    results["h3"] = h3_cluster.run(ld)
    print()
    print(">>> Гипотеза 4")
    results["h4"] = h4_syntax.run(ld)
    print()
    print(">>> Гипотеза 5")
    results["h5"] = h5_selfcite.run(ld)
    print()

    # сохраняем json (без numpy/множеств - сериализуем грубо)
    def default(o):
        try:
            return list(o)
        except Exception:
            return str(o)
    Path("results.json").write_text(json.dumps(results, default=default, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Результаты сохранены в results.json")


if __name__ == "__main__":
    main()
