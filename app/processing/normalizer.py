"""
    Нормализация текста нормативных документо (ГОСТ/СНиП/СП/ПУЭ и т.п.)

    Задачи:
        - Удаление типичных колонтитулова/номеров страниц/дат.
        - Склейка переносов слов
        - Унификация маркеров списков и тире
        - Компактизация пробелова и пустых строк
        - Пользовательские правила для колонтитулов

    Публичный API:
        - clean(text: str, config: ClienConfig | None = None) -> str
"""

from __future__ import annotations
from dataclasses import dataclass, field
from types import TracebackType
from typing import Iterable, List, Pattern

import re
import unicodedata

# =====================================================================
# Конфигурация
# =====================================================================

@dataclass
class CleanConfig:

    # Удаляем строки, которые выглядят как номер страниц
    remove_page_numbers: bool = True

    # Удаляем строки-колонтитулы/шапки/футеры
    remove_headers_footers: bool = True

    # Считать строку колонтитулом, если она встречается не мнее 3х раз
    header_min_repeats: int = 3

    # Максимальная длина строки, чтобы ее рассматривать как потенциальный колонтитул
    header_max_len: int = 120

    # Дополнительные шаблоны колонтитулова от пользователя
    extra_header_footer_regexes: List[str] = field(default_factory=list)

    # Нормировать маркеры списков к виду
    normalize_bullets: bool = True

    # Каким символом начинать маркеты списков при норализации
    bullet_sumbol: str = "-"

    # Сводить более двух пустых строк подряд к двум
    max_consecutive_newlines: int = 2

    # Свободить повторяющиеся пробелы и табы к одному пробелу
    collapse_spaces: bool = True

    # Пробовать склеивать переносы слов
    fix_hyphenation: bool = True

    # Приводить разные тире к одному символу
    normalize_dashes: bool = True

    # Включить легкую скелейку одиночных переносов внутри предложений
    soften_inline_newlines: bool = False

    # Максимальная длина фрагмента строки для soft inline merge
    inline_newline_window: int = 60


# =====================================================================
# Предкомпилированные шаблоны
# =====================================================================

# soft hyphen и различные виды дефисов/тире
_SOFT_HYPHEN = "\u00ad"
_HYPHENS = r"\-\u2010\u2011\u2012\u2013\u2014\u2015"

RE_HYPHEN_NEWLINE = re.compile(rf"(\w)[{_HYPHENS}]\n(\w)")
RE_SOFT_HYPHEN = re.compile(_SOFT_HYPHEN)

# Маркеры списков в начале строки
RE_BULLET_LINE = re.compile(r"^[\s•·\*\-\u2013\u2014]+\s+", re.MULTILINE)

# Разные тире
RE_ANY_DASH = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015]+")

# Номера страниц / служебные строки
RE_PAGE_NUM_PURE = re.compile(r"^\s*[–—\-]?\s*\d{1,4}\s*[–—\-]?\s*$")
RE_PAGE_WORD = re.compile(r"^\s*(стр\.?|страница|page)\s*[№:]?\s*\d{1,4}\s*$", re.IGNORECASE)
RE_DATE_LINE = re.compile(
    r"^\s*(?:\d{1,2}[./]\d{1,2}[./]\d{2,4}|[А-Яа-яA-Za-z]+\s+\d{4}\s*г\.)\s*$"
)

# Подсказки, характерные для колонтитулов норммативных документов
RE_HEADER_HINT = re.compile(
    r"(ГОСТ|СНиП|СП|ПУЭ|СТО)\s+[A-ZА-Я0-9\-.:/]+|Издание|Изм\.\s*\d+|UTD|ISO", re.IGNORECASE,
)

# Служебные шаблоны для "продолжения таблицы", "окончание таблицы"...
RE_TABLE_CONT = re.compile(
    r"^\s*(продолжение|окончание)\s+таблицы\s+\d+([.,].*)?$", re.IGNORECASE
)
RE_TABLE_HEAD = re.compile(r"^\s*таблица\s+\d+([.,].*)?$", re.IGNORECASE)
RE_FIG_HEAD = re.compile(r"^\s*рисунок\s+\d+([.,].*)?$", re.IGNORECASE)


# =====================================================================
# Вспомогательные функции
# =====================================================================

# Удалить неотображаемые управляющие символы, кроме перевода строки и таба
def _strip_control_chars(text: str) -> str:
    return "".join(ch for ch in text if ch in ("\n", "\t") or unicodedata.category(ch)[0] != "C")

# NFKC - нормализация (выравнивает разные формы знаков), удаляем soft hyphen
def _normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = RE_SOFT_HYPHEN.sub("", text)
    return text

# Склейка переносов по дефису / тире на границе строки
def _fix_hyphenation(text: str) -> str:

    # Несколько проходов, чтобы захватить последовательные случаи
    for _ in range(3):
        new_text = RE_HYPHEN_NEWLINE.sub(r"\1\2", text)
        if new_text == text:
            break
        text = new_text
    return text

# Единообразные маркеры списков в начале строки
def _normalize_bullets(text: str, bullet_symbol: str = "-") -> str:

    def repl(match: re.Match) -> str:
        return f"{bullet_symbol} "
    return RE_BULLET_LINE.sub(repl, text)

# Разные виды тире
def _normalize_dashes(text: str) -> str:
    return RE_ANY_DASH.sub("-", text)


# Легкая слейка одиночных переносов строк внутри коротких фрагментов
def _soften_inline_newlines(text: str, window: int = 60) -> str:
    lines = text.split("\n")
    out: List[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if i + 1 < len(lines):
            nxt = lines[i + 1]

            # Если обе строки не слишком длиные и не пустые - склеим через пробел
            if 0 < len(cur) <= window and 0 < len(nxt) <= windows:
                out.append(f"{cur} {nxt}".strip())
                i += 2
                continue
        out.append(cur)
        i += 1
    return "\n".join(out)

def _collapse_spaces_and_newlines(
    text: str, collapse_spaces: bool, max_consecutive_newlines: int 
) -> str:
    if collapse_spaces:

        # Табы и пробелы сводим к одному пробелу
        text = text.replace("\t", " ")
        text = re.sub(r"[ ]{2,}", " ", text)

    # Ограничим последовательные пусты строки
    if max_consecutive_newlines >= 1:

        # Превращаем последовательности \n в максимум N
        pattern = r"\n{" + str(max_consecutive_newlines + 1) + r",}"
        replacement = "\n" * max_consecutive_newlines
        text = re.sub(pattern, replacement, text)

    return text.strip()

# Удаление колонтитулов/дат/номеров страниц
def _drop_header_footer_lines(
    lines: List[str],
    cfg: CleanConfig,
    extra_patterns: Iterable[Pattern] | None = None,
) -> List[str]:

    # Подготовим пользователские паттерны
    extra_compiled: List[Pattern] = list(extra_patterns or [])
    for pat in cfg.extra_header_footer_regexes:
        try:
            extra_compiled.append(re.compile(pat, re.IGNORECASE))
        except re.error:

            # Игнорируем некорректные regex
            pass
    
    # Подсчет повторов коротких строк
    freq: dict[str, int] = {}
    for raw in lines:
        s = raw.strip()
        if not s or len(s) > cfg.header_max_len:
            continue
        if RE_HEADER_HINT.search(s) or RE_PAGE_WORD.match(s) or RE_PAGE_NUM_PURE.match(s):
            freq[s] = freq.get(s, 0) + 1

    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            out.append(ln) # Пустые строки сохраняем
            continue

        # Шаг 1. Явные номера страниц/строки - даты
        if cfg.remove_page_numbers and (RE_PAGE_NUM_PURE.match(s) or RE_PAGE_WORD.match(s)):
            continue
        if RE_DATE_LINE.match(s):
            continue

        # Шаг 2. Табличные служебные  строки
        if RE_TABLE_CONT.match(s):
            continue

        # Шаг 3. Эвристика колонтитула
        is_headerish = RE_HEADER_HINT.search(s) or RE_TABLE_HEAD.match(s) or RE_FIG_HEAD.match(s)
        repeated_enough = freq.get(s, 0) >= cfg.header_min_repeats
        if cfg.remove_headers_footers and is_headerish and (repeated_enough or len(s) <= cfg.header_max_len):
            
            # Если строка короткая и похожа на колонтитул - удаляем
            continue

        # Шаг 4. Пользовательские паттерны
        if any(p.search(s) for p in extra_compiled):
            continue

        out.append(ln)

    return out


# =======================================================================
# Основная функция
# =======================================================================

# Полная нормализация текста
def clean(text: str, config: CleanConfig | None = None) -> str:
    cfg = config or CleanConfig()

    # Унификация юникода и удаления скрытых символов
    text = _normalize_unicode(text)
    text = _strip_control_chars(text)

    # Склейка переносов слов и нормализация тире/маркеров списков
    if cfg.fix_hyphenation:
        text = _fix_hyphenation(text)
    if cfg.normalize_dashes:
        text = _normalize_dashes(text)
    if cfg.normalize_bullets:
        text = _normalize_bullets(text, cfg.bullet_symbol)

    # Удаление колонтитулов/дат/номеров страниц
    lines = text.splitlines()
    lines = _drop_header_footer_lines(lines, cfg)

    text = "\n".join(lines)

    # Осторожная склейка коротких строк
    if cfg.soften_inline_newlines:
        text = _soften_inline_newlines(text, cfg.inline_newline_window)

    # Компактизация пробелов и пустых строк
    text = _collapse_spaces_and_newlines(
        text, collapse_spaces=cfg.collapse_spaces, max_consecutive_newlines=cfg.max_consecutive_newlines
    )
    return text

# Внутрення обертка с защитой (чтобы не склеить большие абзацы по ошибке)

# Обертка над _soften_inline_newlines с проверками
def _soffen_inline_newlines_quard(text: str, window: int) -> str:
    lines = text.split("\n")
    out: List[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            if (
                0 < len(cur) <= window
                and 0 < len(nxt) <= window
                and not re.search(r"[.!?…:;]\s*$", cur)
            ):
                out.append(f"{cur} {nxt}".strip())
                i += 2
                continue
        out.append(cur)
        i += 1
    return "\n".join(out)

__all__ = ["CleanConfig", "clean"]