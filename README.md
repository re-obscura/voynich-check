# voynich-check

> Манускрипт Войнича после восьми частей исследования: пять гипотез, маленькая языковая модель, четыре атаки на дешифровку, шесть углублённых статистических тестов и анализ внешних факторов — всё сходится к одной картине.

Независимое воспроизводимое исследование: берём EVA-транскрипцию Такахаши (IVTFF 2.0 с voynich.nu) и прогоняем через рукопись растущую батарею проверок — от классических гипотез до маленькой обученной LLM и поиска внешнего ключа. В каждой проверке есть контрольная проба — то, что обычно пропускают, обрадовавшись первому красивому числу.

📖 **Хотите почитать как историю?** Откройте [`LONGREAD.md`](./LONGREAD.md) — единый лонгрид в восемь актов, охватывающий весь путь живым языком с поворотами и финальной таблицей. Технические детали каждой части — в отдельных документах ниже.

**Финальный итог (см. [`CONCLUSION.md`](./CONCLUSION.md)):** рукопись — профессионально исполненная в начале XV в. имитация письменности, порождённая локальной слоговой генерацией (порядок памяти 1–2 глифа), с переключением между двумя генеративными режимами (Currier A/B), без семантики (верхняя граница смысла ≤0.02 бит/глиф, в 75× ниже языка) и без шифра/ключа к известным языкам.

---

## Карта исследования (читать по порядку)

| Документ | О чём |
|---|---|
| [`LONGREAD.md`](./LONGREAD.md) | **Единый лонгрид в восемь актов** — весь путь одним живым текстом. Если хотите почитать как историю, начните отсюда |
| [`ARTICLE.md`](./ARTICLE.md) | Часть 1: пять исходных гипотез — ни одна не пережила контрольный тест |
| [`ARTICLE_LM.md`](./ARTICLE_LM.md) | Часть 2: маленькая glyph-level LLM — энтропия, обобщение, генеративный тест, пробы, 3D-карта памяти |
| [`DECODE.md`](./DECODE.md) | Часть 3: четыре атаки на дешифровку — LM-подбор шифра, именование, Procrustes, reverse-LM |
| [`ROUND4.md`](./ROUND4.md) | Часть 4: классика — Ципф, Heaps, long-range MI, Марков, позиционный профиль, PPMI, temporal drift |
| [`ROUND5.md`](./ROUND5.md) | Часть 5: текст как физический объект — word-wrap reconstruction, PACF, generative sufficiency (3-грамма) |
| [`ROUND6.md`](./ROUND6.md) | Часть 6: глубже — residual map, слоговая грамматика, information bound, bio-B атака |
| [`ROUND7.md`](./ROUND7.md) | Часть 7: поиск внешнего ключа — номенклатор-тест, кросс-языковая perplexity (латынь/итальянский/английский) |
| [`EXTERNAL.md`](./EXTERNAL.md) | Часть 8: внешние факторы — C14, чернила, провенанс, писцы, кодикология, маргиналии |
| [`CONCLUSION.md`](./CONCLUSION.md) | **Синтез всех восьми частей + таблица всех ~30 тестов + финальная интерпретация** |

---

## Ключевые числа (из независимой репликации)

Корпус: **183 фолио · 4183 строки · 31 005 токенов · 7181 уникальное слово · 22 глифа**.

### Часть 1 — пять гипотез (ARTICLE.md)

| Метрика | Значение | Что значит |
|---|---|---|
| Суффикс `-am`: конец строки vs середина | **11.5% vs 0.60%** | Line-edge эффект в 19× сильнее |
| 4-граммы на ≥3 тетрадях (bio/recipes/herbal) | **0 / 0 / 0** | Шаблонной генерации нет |
| Currier A/B: silhouette / ARI | **+0.041 / 0.954** | Единственное деление, что кластеризуется |
| LOO униграммы vs биграммы | **89.3% vs 69.5%** | Порядок слов не несёт информации |
| Self-citation: Войнич vs контрольные | **0.97× vs 1.34–1.91×** | Слабее всех живых языков |

### Часть 2 — LM-анализ (ARTICLE_LM.md)

| Метрика | Значение | Что значит |
|---|---|---|
| CE(ctx=256) test: Войнич vs живые языки | **1.77 vs 1.88–2.25** бит/глиф | Войнич предсказуемее — но см. ниже (нормировка) |
| Редундансность R = 1−H/H_max (нормировка на алфавит) | **60.9% ≈ Библия 60.8%** | «Сверхпредсказуемость» — артефакт малого алфавита |
| Зазор обобщения: Войнич vs живые | **0.09 vs 0.44–0.56** (z=−8.3) | Войнич обобщает лучше языков — признак ограниченной модели генерации |
| Cross-dialect CE asymmetry | **+0.36 бит/глиф** | Currier A/B — разные генеративные машины |

### Часть 5 — генеративная модель (ROUND5.md, ключевой результат)

| Метрика | Значение | Что значит |
|---|---|---|
| **3-грамма глифов воспроизводит статистику** | **ошибка 0.9%** | Минимальная достаточная модель имитации |
| PACF порядок памяти | 1–2 глифа | Короткая слоговая память |
| Word-wrap reconstruction | не работает (CV=0.36) | Нюанс H1: границы строк — не жадный перенос |

### Часть 6 — граница смысла (ROUND6.md, ключевой результат)

| Метрика | Значение | Что значит |
|---|---|---|
| **Information bound** (CE_3gram − CE_LM) | **0.02 бит/глиф** | Максимум «смысла» — в 75× ниже языка |
| Residual map (между/внутри фолио) | 0.10 | Остаток случаен — нет «трудных» страниц |
| Topic signatures (фолио с подписями) | 0 из 183 | Гипотеза именования закрыта |

### Часть 8 — внешние факторы (EXTERNAL.md)

| Факт | Значение |
|---|---|
| C14 пергамента (4 образца) | 1434 ± 18, 95% CI **1404–1435** (однородный) |
| Чернила | Iron gall, один для текста и рисунков; анахронизмов нет |
| Провенанс | Документирован с 1639 (Бареш, Прага) |

---

## Быстрый старт

```bash
git clone https://github.com/re-obscura/voynich-check.git
cd voynich-check

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python fetch_data.py        # качает транскрипцию + 4 контрольных текста в ./data/
python -m voi.run_all       # прогоняет 5 исходных гипотез, пишет results.json
```

### Воспроизведение частей 2–8 (LM и углублённый анализ)

```bash
# Часть 2: обучить семейство glyph-LM + анализы A–G
python -m voynich_lm.run_experiments          # ~25 мин на GPU

# Части 4–7: углублённые анализы (CPU, быстро)
python -m voynich_lm.analyze.round4           # Ципф/MI/Марков/PPMI/drift
python -m voynich_lm.analyze.round5           # word-wrap/PACF/3-грамма
python -m voynich_lm.analyze.round6           # residual/info-bound/signatures
python -m voynich_lm.analyze.round7           # внешний ключ (нужны data/latin_combined.txt, data/italian.txt)
```

Для части 7 (кросс-языковая) нужны тексты-кандидаты: латынь (Цезарь/Цицерон/Вергилий/Ливий с thelatinlibrary.com) и итальянский (Данте, Gutenberg #1012). См. раздел источников ниже.

### Отдельные модули

Каждую гипотезу и анализ можно прогнать независимо:

```bash
python -m voi.entropy        # матчасть + условная энтропия H₂|₁
python -m voi.h1_position    # гипотеза 1: позиция в абзаце / line-edge
python -m voi.h2_ngrams      # гипотеза 2: шаблонные блоки
python -m voi.h3_cluster     # гипотеза 3: кластеризация Currier/секция/писец
python -m voi.h4_syntax      # гипотеза 4: синтаксис (би- vs униграммы)
python -m voi.h5_selfcite    # гипотеза 5: самоцитирование Тимм/Шиннер
```

---

## Структура проекта

```
voynich-check/
├── ARTICLE.md            # часть 1: пять гипотез
├── ARTICLE_LM.md         # часть 2: LM-анализ
├── DECODE.md             # часть 3: атаки на дешифровку
├── ROUND4.md             # часть 4: классические тесты
├── ROUND5.md             # часть 5: физический объект
├── ROUND6.md             # часть 6: остаток и граница смысла
├── ROUND7.md             # часть 7: внешний ключ
├── EXTERNAL.md           # часть 8: внешние факторы
├── CONCLUSION.md         # синтез всех частей
├── fetch_data.py         # авто-загрузка данных (voynich.nu + Gutenberg)
├── requirements.txt
├── LICENSE
├── voi/                  # часть 1: пять гипотез
│   ├── common.py         # JSD, TF-IDF, n-граммы, helpers
│   ├── parse_ivtff.py    # парсер IVTFF → размеченный датасет строк
│   ├── entropy.py        # матчасть + H₂|₁
│   ├── h1_position.py … h5_selfcite.py
│   └── run_all.py        # сводный прогон
└── voynich_lm/           # части 2–7: LM + углублённый анализ
    ├── data.py           # glyph-токенизатор, потоки, hold-out по фолио
    ├── model.py          # nanoGPT-декодер (RoPE, causal attention)
    ├── train.py          # тренер + early-stopping
    ├── perplexity.py     # кросс-энтропия в бит/глиф
    ├── run_experiments.py# оркестратор частей A–G
    └── analyze/          # 16 аналитических модулей
        ├── entropy_generalization.py   # часть 2-A
        ├── generative.py               # часть 2-B
        ├── cross_dialect.py            # часть 2-D
        ├── position.py                 # часть 2-E
        ├── probing.py                  # часть 2-F
        ├── embeddings.py               # часть 2-G (3D-карта памяти)
        ├── decode_cipher.py            # часть 3.1
        ├── decode_naming.py            # часть 3.2
        ├── decode_alignment.py         # часть 3.3
        ├── reverse_attention.py        # часть 3.4
        ├── round4.py                   # часть 4 (6 тестов)
        ├── round5.py                   # часть 5 (word-wrap/PACF/3-gram)
        ├── round6.py                   # часть 6 (residual/info-bound)
        └── round7.py                   # часть 7 (внешний ключ)
```

Графики всех частей — в `figures/` (включая вращаемую 3D-карту памяти `G_3d_memory.html`), чекпоинты обученных моделей — в `checkpoints/`, результаты — в `results*.json` и `round*_results.json`.

---

## Источники данных

| Файл | Откуда | Что это |
|---|---|---|
| `IT2a-n.txt` | [voynich.nu/data/IT2a-n.txt](https://www.voynich.nu/data/IT2a-n.txt) | Транскрипция Такахаши, EVA, IVTFF 2.0 |
| `kjv.txt` | [Gutenberg #10](https://www.gutenberg.org/ebooks/10) | Библия (King James Version) |
| `moby.txt` | [Gutenberg #2701](https://www.gutenberg.org/ebooks/2701) | Moby Dick, Г. Мелвилл |
| `paradise.txt` | [Gutenberg #20](https://www.gutenberg.org/ebooks/20) | Paradise Lost, Дж. Мильтон |
| `caesar.txt` | [Gutenberg #1522](https://www.gutenberg.org/ebooks/1522) | Julius Caesar, У. Шекспир |
| `latin_combined.txt` | [thelatinlibrary.com](http://www.thelatinlibrary.com/) | Цезарь/Цицерон/Вергилий/Ливий (для части 7) |
| `italian.txt` | [Gutenberg #1012](https://www.gutenberg.org/ebooks/1012) | Данте, Divina Commedia (оригинал, для части 7) |

Формат транскрипции: [IVTFF specification (PDF)](https://www.voynich.nu/software/ivtt/IVTFF_format.pdf). Внутри файла inline закодированы метаданные каждой страницы: тетрадь (`$Q`), тип иллюстрации → секция (`$I`), диалект Currier A/B (`$L`), писец по Davis (`$H`).

---

## Методология

- **Что считается** — конкретный, воспроизводимый показатель для каждой проверки.
- **Контроль** — Монте-Карло (1000–2000 итераций), перестановочные тесты, сравнение с живыми языками и shuffle-null.
- **Метрики:** JSD, TF-IDF + косинус, silhouette, ARI, Манн–Уитни, MultinomialNB LOO, UMAP + K-means, кросс-энтропия glyph-LM, PACF, simulated annealing, Procrustes, PPMI+SVD, information bound.
- **Hold-out на уровне фолио** (не позиций), early-stopping для LM — критично для малого корпуса.

Точные значения чувствительны к составу корпуса и параметрам. Решения принимаются по **направлению эффекта и порядку величины**, а не по третьему знаку после запятой. Полная таблица всех ~30 тестов — в [`CONCLUSION.md`](./CONCLUSION.md).

---

## Благодарности

- [Rene Zandbergen (voynich.nu)](https://www.voynich.nu/) — за каноническую транскрипцию и формат IVTFF.
- [Takeshi Takahashi](https://www.voynich.nu/transcr.html) — за саму транскрипцию.
- [Project Gutenberg](https://www.gutenberg.org/) и [The Latin Library](http://www.thelatinlibrary.com/) — за контрольные тексты.
- [Yale Beinecke Library](https://beinecke.library.yale.edu/beinecke/collections/beinecke-cipher-voynich-manuscript) — за открытый доступ к факсимиле MS 408.
- [University of Arizona](https://voynich.nu/extra/carbon.html) (Hodgins, 2009) и [McCrone Associates](https://ciphermysteries.com/2011/06/01/voynich-the-mccrone-report-now-online) (2009) — за физический анализ.

## Лицензия

Код — [MIT](./LICENSE). Данные принадлежат их правообладателям (voynich.nu, Project Gutenberg, thelatinlibrary.com) и не входят в репозиторий.
