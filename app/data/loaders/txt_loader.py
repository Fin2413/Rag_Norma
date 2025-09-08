"""
    Загрузчик текстовых регламентов (TXT/MD)

    Публичный API:
        - extract_text_file(path: str) -> str
        - Возвращает сырой текст документа (с легкой нормализацией переносов и,
        при .md, упрощением)

    Примечание:
        - Пытаемся корректно прочитать файл с несколькмики кодировками
        - Markdown - файлы: убираем front matter, кодовые блоки, изображения
"""

from __future__ import annotations
from typing import Iterable

import pathlib
import re

_DEF_CODECS: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1251", "koi8-r", "latin1")

def _read_text_any(path: str | pathlib.Path, encodings: Iterable[str] = _DEF_CODECS) -> str:
    last_err: Exception | None = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception as e:
            last_err = e
            continue

    # fallback: permissive
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# ===========================================================================================
# Упрощение Markdown (Легкое)
# ===========================================================================================

_RE_YAML_FRONT = re.compile(r"(?s)^\s*---\s*\n.*?\n---\s*\n")
_RE_CODE_FENCE = re.compile(r"(?m)^```.*?$[\s\S]*?^```$")  # блоки ```...```
_RE_INLINE_CODE = re.compile(r"`([^`]+)`")
_RE_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)")           # ![alt](url)
_RE_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")           # [text](url) -> text
_RE_BOLD = re.compile(r"(\*\*|__)(.*?)\1")
_RE_ITAL = re.compile(r"(\*|_)(.*?)\1")
_RE_HTML_TAG = re.compile(r"</?[^>]+>")
_RE_MD_HEAD = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
_RE_MD_LIST = re.compile(r"(?m)^\s{0,3}([-*+]|(\d+)[.)])\s+")


def _simplify_markdown(text: str) -> str:

    # Удалить YAML front matter
    text = _RE_YAML_FRONT.sub("", text)
    
    # Удалить блоки кода и инлайн-код - оставим только соержимое инлайн-кода
    text = _RE_CODE_FENCE.sub("", text)
    text = _RE_INLINE_CODE.sub(r"\1", text)

    # Изображения полностью убрать, ссылки - оставить текста
    text = _RE_IMAGE.sub("", text)
    text = _RE_LINK.sub("", text)

    # Форматирование Жирный/Курсив снять
    text = _RE_BOLD.sub(r"\2", text)
    text = _RE_ITAL.sub(r"\2", text)

    # HTML-теги удалить
    text = _RE_HTML_TAG.sub("", text)

    # Заголовки '#' убрать, маркеры списков заменить на тире
    text = _RE_MD_HEAD.sub("", text)
    text = _RE_MD_LIST.sub("- ", text)

    return text

def extract_text_file(path: str) -> str:
    p = pathlib.Path(path)
    raw = _read_text_any(p)

    # Нормализуем переносы строк
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    if p.suffix.lower() == ".md":
        raw = _simplify_markdown(raw)

    # Уберем скрытые нулевые символы/непечатные суррогаты, но не трогаем \n и \t
    raw = "".join(ch for ch in raw if ch =="\n" or ch == "\t" or (ord(ch) >= 32))

    # Трим по краям - глубокая очистка дальше в normalizer.clean()
    return raw.strip()