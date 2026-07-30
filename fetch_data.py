"""
Автоматическая загрузка данных, нужных для воспроизведения.

Запускать из корня проекта:
    python fetch_data.py

Скачивает в ./data/:
  - IT2a-n.txt       транскрипция Такахаши (EVA, IVTFF 2.0) с voynich.nu
  - kjv.txt          Библия (KJV) — Project Gutenberg #10
  - moby.txt         Moby Dick — Project Gutenberg #2701
  - paradise.txt     Paradise Lost — Project Gutenberg #20
  - caesar.txt       Julius Caesar (Шекспир) — Project Gutenberg #1522

Файлы не входят в репозиторий (см. .gitignore / data/), поэтому первый
запуск любого анализа их подтянет. Повторно ничего не качается.
"""
from __future__ import annotations
import sys
import urllib.request
from pathlib import Path

from voi.common import DATA_DIR, DATA_FILES

SOURCES = {
    "voynich": "https://www.voynich.nu/data/IT2a-n.txt",
    "kjv":     "https://www.gutenberg.org/cache/epub/10/pg10.txt",
    "moby":    "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    "paradise": "https://www.gutenberg.org/cache/epub/20/pg20.txt",
    "caesar":  "https://www.gutenberg.org/cache/epub/1522/pg1522.txt",
}


def humanize(n: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def download(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ✓ {dest.name:16s} уже есть ({humanize(dest.stat().st_size)})")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ {dest.name:16s} <- {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "voynich-check/fetch_data"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as fh:
        fh.write(r.read())
    print(f"     скачано {humanize(dest.stat().st_size)}")
    return True


def main() -> int:
    print("Загрузка данных в ./data/")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for key, url in SOURCES.items():
        try:
            download(url, DATA_FILES[key])
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {key}: {exc}", file=sys.stderr)
            return 1
    print("Готово. Можно запускать: python -m voi.run_all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
