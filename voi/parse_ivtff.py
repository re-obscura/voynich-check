"""
Парсер IVTFF-транскрипции Такахаши (EVA) -> размеченный датасет строк.

Формат IVTFF 2.0 (файл IT2a-n.txt):
  Строки-заголовки страницы:
      <f1r>      <! $Q=A $P=A $F=a $B=1 $I=T $L=A $H=1 $C=1 $X=V>
  Строки-локусы (текст рукописи):
      <f1r.1,@P0>       <%>fachys.ykal.ar.ataiin...

Переменные страницы:
  $Q  - тетрадь (A=1..T=20)
  $I  - тип иллюстрации: H=herbal, B=biological, Z=zodiac(astro),
        C=cosmological, P=pharmaceutical, S=stars/astro-recipes, T=text-only
  $L  - диалект Currier (A/B)
  $H  - писец по Davis (1..5)
  $C  - codicological hand (Currier)

Спецмаркеры в тексте локуса:
  <%> - отступ / начало абзаца (параграф начинается с этой строки)
  <$> - перенос слова на следующую строку (line-fill)
  <-> - соединение слов (часто висячий дефис) - здесь это часть глифов
  .   - разделитель слов в EVA
  ?   - нераспознанный/неуверенный глиф (заменяем на пустой - опускаем)

Локус-тип (после запятой): первый символ - маркер межстрочного переноса:
  @ - обычная строка (строка начинается с этой записи)
  + - продолжение предыдущей строки без видимого разрыва (потоковая)
  = - последняя строка абзаца (последний параграф / замыкающий)
  * - строка с абзацным отступом (<%> marker)
  & - ...

Нас интересует главным образом параграфный текст P* (running text),
с наблюдаемой позицией в абзаце (первая/последняя/середина) и позицией
на странице (верх/низ).
"""
from __future__ import annotations
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

from voi import common

import pandas as pd

# --- mapping illustration type -> тематическая секция статьи ---
ILLUS_TO_SECTION = {
    "H": "herbal",
    "B": "bio",
    "Z": "astro",
    "C": "cosmo",
    "P": "pharma",
    "S": "recipes",   # S=stars/astro-labels, но в статье "recipes" = текст со звёздочками на полях (Q20)
    "T": "text",
}

# регулярные выражения
RE_PAGE = re.compile(r"<f(\d+)([rv])>\s*<!\s*([^>]*)>")
RE_LOCUS = re.compile(r"<f(\d+)([rv])\.(\d+),([^>]*)>\s*(.*)")
RE_VARS = re.compile(r"\$(.)=([^ ]+)")


def _folio_id(num: str, side: str) -> str:
    return f"f{num}{side}"


def _folio_num(folio: str) -> int:
    m = re.match(r"f(\d+)", folio)
    return int(m.group(1))


def clean_eva_text(raw: str) -> str:
    """Удалить inline-маркеры IVTFF из текста локуса, оставить чистый EVA-текст."""
    s = raw
    # удалить <%>  <*>  <$>  <!> ... встроенные теги
    s = re.sub(r"<[^>]*>", "", s)
    # удалить висячие пробелы
    return s.strip()


def eva_tokens(text: str) -> list[str]:
    """
    EVA-слова разделяются точкой. '?' = нечитаемый глиф - убираем как отдельный
    элемент, но сохраняем в слове, если он часть слова (заменим на '' для чистоты
    частотного анализа глифов).
    """
    s = clean_eva_text(text)
    if not s:
        return []
    toks = [t for t in s.split(".") if t != ""]
    # убираем нечитаемые/нечёткие символы '?'
    toks = [t.replace("?", "") for t in toks]
    toks = [t for t in toks if t != ""]
    return toks


def illus_to_section(illus: str, quire: str) -> str:
    """
    Маппинг типа иллюстрации -> тематическая секция статьи.
    Особый случай: $I=S (звёзды/звёздочки) в тетради T (Q20, f103-116) -
    это секция 'recipes' (текст со звёздочками на полях);
    вне Q20 это астрологическая звёздная диаграмма -> 'astro'.
    """
    if illus == "S":
        return "recipes" if quire == "T" else "astro"
    return ILLUS_TO_SECTION.get(illus, "?")



@dataclass
class Line:
    folio: str
    folio_num: int
    locus_seq: int            # номер локуса на странице
    cont_marker: str          # '@','+','=','*','&','~' - межстрочный маркер
    locus_type: str           # 'P0','P1','Lc',... (P = параграфный текст)
    raw_text: str
    tokens: list[str] = field(default_factory=list)
    # заполнится потом
    section: str = ""
    quire: str = ""
    currier: str = ""         # $L
    scribe: str = ""          # $H
    illus: str = ""           # $I
    page_line_index: int = 0  # порядковый номер строки-локуса на странице (0-based среди текстовых)
    page_n_lines: int = 0
    para_start: bool = False  # начинается ли абзац с этой строки (<%> marker)
    para_index: int = 0       # индекс абзаца на странице
    para_pos: int = 0         # позиция в абзаце (0=первая, 1=последняя, 2=середина)
    top_half: Optional[bool] = None  # верх/низ страницы


def parse_ivtff(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Возвращает (lines_df, folios_df).
      lines_df  - по одной строке на локус-строку параграфного текста
      folios_df - по одной строке на фолио с метаданными страницы
    """
    lines: list[Line] = []
    folios: list[dict] = []

    # текущие метаданные страницы
    cur: dict = {}

    # сначала соберём все локусы постранично, потом проставим позиции
    page_lines: dict[str, list[Line]] = {}

    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if not ln.strip():
                continue
            # заголовок страницы
            m = RE_PAGE.match(ln)
            if m and "." not in ln.split(">", 1)[0]:
                num, side, varsblock = m.group(1), m.group(2), m.group(3)
                d = dict(RE_VARS.findall(varsblock))
                folio = _folio_id(num, side)
                cur = {
                    "folio": folio,
                    "folio_num": _folio_num(folio),
                    "quire": d.get("Q", "?"),
                    "illus": d.get("I", "?"),
                    "currier": d.get("L", "?"),
                    "scribe": d.get("H", "?"),
                    "section": illus_to_section(d.get("I", "?"), d.get("Q", "?")),
                }
                folios.append(cur.copy())
                page_lines.setdefault(folio, [])
                continue

            m = RE_LOCUS.match(ln)
            if m:
                num, side, seq, typ, txt = m.groups()
                folio = _folio_id(num, side)
                if not cur or cur.get("folio") != folio:
                    # локус без предшествующего заголовка (редко) - пропустим мету
                    cur = {"folio": folio, "folio_num": _folio_num(folio),
                           "quire": "?", "illus": "?", "currier": "?", "scribe": "?",
                           "section": illus_to_section("?", "?")}
                cont = typ[0]
                ltype = typ[1:]
                para_start = "<%>" in ln
                L = Line(
                    folio=folio,
                    folio_num=_folio_num(folio),
                    locus_seq=int(seq),
                    cont_marker=cont,
                    locus_type=ltype,
                    raw_text=clean_eva_text(txt),
                    tokens=eva_tokens(txt),
                    section=cur["section"],
                    quire=cur["quire"],
                    currier=cur["currier"],
                    scribe=cur["scribe"],
                    illus=cur["illus"],
                    para_start=para_start,
                )
                lines.append(L)
                page_lines.setdefault(folio, []).append(L)

    # проставить позицию на странице и в абзаце
    for folio, llist in page_lines.items():
        n = len(llist)
        # сначала отфильтруем только содержащие текст параграфного типа (P*)
        text_lines = [L for L in llist if L.tokens and L.locus_type.startswith("P")]
        nt = len(text_lines)
        # номера в page_line_index: по всем строкам, но top/bottom считаем по текстовым
        for idx, L in enumerate(text_lines):
            L.page_line_index = idx
            L.page_n_lines = nt
            L.top_half = idx < (nt / 2)

    # позиция в абзаце
    for folio, llist in page_lines.items():
        text_lines = [L for L in llist if L.tokens and L.locus_type.startswith("P")]
        # абзацы отделяются <%> (para_start). Абзац = группа подряд идущих строк.
        para_groups: list[list[Line]] = []
        cur_group: list[Line] = []
        for L in text_lines:
            if L.para_start and cur_group:
                para_groups.append(cur_group)
                cur_group = []
            cur_group.append(L)
        if cur_group:
            para_groups.append(cur_group)
        for pi, grp in enumerate(para_groups):
            for li, L in enumerate(grp):
                L.para_index = pi
                if len(grp) == 1:
                    L.para_pos = 0  # и первая и последняя - отнесём к первой
                elif li == 0:
                    L.para_pos = 0
                elif li == len(grp) - 1:
                    L.para_pos = 1
                else:
                    L.para_pos = 2

    cols = ["folio", "folio_num", "locus_seq", "cont_marker", "locus_type",
            "section", "quire", "currier", "scribe", "illus",
            "page_line_index", "page_n_lines", "para_index", "para_pos",
            "para_start", "top_half",
            "raw_text", "tokens"]
    data = []
    for L in lines:
        data.append({c: getattr(L, c) for c in cols})
        # tokens - список, оставим как объект
    lines_df = pd.DataFrame(data)
    folios_df = pd.DataFrame(folios)
    return lines_df, folios_df


def add_n_token_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Удобные столбцы: n_tokens, токены строкой."""
    df = df.copy()
    df["n_tokens"] = df["tokens"].apply(len)
    df["tokens_str"] = df["tokens"].apply(lambda t: " ".join(t))
    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else str(common.data_path("voynich"))
    lines_df, folios_df = parse_ivtff(path)
    lines_df = add_n_token_columns(lines_df)
    print("=== FOLIOS ===")
    print(folios_df.shape)
    print(folios_df["section"].value_counts())
    print("currier:", folios_df["currier"].value_counts().to_dict())
    print("scribe:", folios_df["scribe"].value_counts().to_dict())
    print()
    print("=== LINES (paragraph text only) ===")
    # все текстовые локусы (P, L, C, R) с непустыми токенами
    p = lines_df[lines_df["n_tokens"] > 0]
    print("total text-lines:", len(p))
    print("total tokens:", int(p["n_tokens"].sum()))
    alltok = [t for toks in p["tokens"] for t in toks]
    print("unique tokens:", len(set(alltok)))
    import collections
    chars = collections.Counter()
    for t in alltok:
        chars.update(t)
    print("distinct glyphs (chars):", len(chars), "top:", chars.most_common(8))
    print()
    print("by section:")
    print(p.groupby("section")["n_tokens"].agg(["count", "sum"]))
