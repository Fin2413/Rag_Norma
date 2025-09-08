"""
    Разработка структуры нормативного текста

    Функции:
        - find_anchors(text) -> List[Anchor] извлекает якори заголовоков
        - annotate_chunks_with_sections(text, chunks) -> List[str]: для каждого чанка возвращает метку ближашего
        - build_outline(text) -> List[Anchor]: плоский список якорей с уровнями
    
    Зависимости: только стандартная библиотека
"""

from __future__ import annotations
from dataclasses import dataclass
from multiprocessing import current_process
from typing import List, Optional, Tuple
import re

from app.cli import ingest

# =====================================================================
# Регулярные выражения
# =====================================================================

# Разделы верхнего уровня
RE_SECTION = re.compile(
    r"^\s*(?P<kind>Раздел|Глава|Часть)\s+(?P<id>(?:[IVXLCDM]+|\d+))\s*[.)]?\s*(?P<title>.*)$",
    re.IGNORECASE | re.MULTILINE,
)

# Приложения (часто буквенные индексы)
RE_APPENDIX = re.compile(
    r"^\s*(?P<kind>Приложение)\s*(?P<id>[A-ЯA-Z\d]+)?\s*[.)]?\s*(?P<title>.*)$",
    re.IGNORECASE | re.MULTILINE,
)

# Пункты/Подпункты (в т.ч. краткая форма)
RE_CLAUSE = re.compile(
    r"^\s*(?P<kind>Пункт|Подпункт|П\.|п\.)\s+(?P<id>\d+(?:\.\d+){0,4})\s*[.)]?\s*(?P<title>.*)$",
    re.IGNORECASE | re.MULTILINE,
)

# Нумерованные заголовки без ключевого слова
RE_NUM_TITLE = re.compile(
    r"^\s*(?P<id>\d+(?:\.\d+){0,2})\s+(?P<title>[A-ЯA-ZЁ].{2,})$",
    re.MULTILINE,
)

# Таблицы/рисунки - полезно для навигации, но обычно Не верний уровень секционирования
RE_TABLE = re.compile(
    r"^\s*(?P<kind>Таблица)\s+(?P<id>\d+(?:\.\d+)*)\s*(?P<title>.*)$",
    re.IGNORECASE | re.MULTILINE, 
)

RE_FIG = re.compile(
    r"^\s*(?P<kind>Рисунок)\s+(?P<id>\d+(?:\.\d+)*)\s*(?P<title>.*)$",
    re.IGNORECASE | re.MULTILINE,
)


# =======================================================================
# Структуры данных
# =======================================================================

@dataclass
class Anchor:
    kind: str                           # 'section' | 'appendix' | 'clause' | 'table' | 'figure'
    id: str                             # индетификатор (номер/буква/римская цифра)
    title: str                          # текст поле заголовка (может быть пустым)
    start: int                          # смещение  начала заголовка в исходном тексте
    end: int                            # смещение конца области (границы до следующего анкера)
    level: int                          # иерархический уровень
    raw_kind: Optional[str] = None      # исходное слово из текста (раздел/глава/пунк)

# Человекочитаемая метка
def label(self) -> str:
    if self.kind == "section":
        base = f"{self.raw_kind or 'Раздел'} {self.id}"
    elif self.kind == "appendix":
        base = f"Приложение {self.id or ''}".strip()
    elif self.kind == "clause":
        base = f"Пункт {self.id}"
    elif self.kind == "numtitle":
        base = f"{self.id}"
    elif self.kind == "table":
        base = f"Таблица {self.id}"
    elif self.kind == "figure":
        base = f"Рисунок {self.id}"
    else:
        base = self.id
    if self.title:
        return f"{base}: {self.title}".strip()
    return base


# ========================================================================
# Вспомогательные функции
# ========================================================================

# Грубая оценка уровня по количеству точек в идентификаторе
def _bot_depth(num_id: str) -> int:
    return num_id.count(".") + 1 if num_id else 1

def _mk_anchor(kind: str, raw_kind: str, _id: str, title: str, start: int) -> Anchor:
    k = kind

    # Уровни: section\appendix = 1; clause\numtitle - по глубине; table\figure - 99
    if k in ("section", "appendix"):
        level = 1
    elif k in ("clause", "numtitle"):
        level = min(1 + _dot_depth(id), 6)
    elif k in ("table", "figure"):
        level = 99
    else:
        level = 5
    return Anchor(kind=k, raw_kind=raw_kind, id=_id.strip(), title=(title or "").strip(), start=start, end=1, level=level)


# ========================================================================
# Извлечение якорей
# ========================================================================

# Находит заголовки и возвращает список Anchor, отсортированные по позиции
# Поля end вычисляются как начало следующего анкера

def find_anchors(text: str) -> List[Anchor]:
    anchors: List [Anchor] = []

    for m in RE_SECTION.finditer(text):
        anchors.append(_mk_anchor("section", m.group("kind"), m.group("id"), m.group("title"), m.start()))

    for m in RE_APPENDIX.finditer(text):
        anchors.append(_mk_anchor("appendix", m.group("kind"), m.group("id") or "", m.group("title"), m.start()))

    for m in RE_CLAUSE.finditer(text):
        anchors.append(_mk_anchor("clause", m.group("kind"), m.group("id"), m.group("title"),m.start()))

    for m in RE_NUM_TITLE.finditer(text):
        anchors.append(_mk_anchor("numtitle", m.group("kind"), m.group("id"),m.group("title"), m.start()))

    for m in RE_TABLE.finditer(text):
        anchors.append(_mk_anchor("table", m.group("kind"), m.group("id"), m.group("title"), m.start()))

    for m in RE_FIG.finditer(text):
        anchors.append(_mk_anchor("figure", m.group("kind"), m.group("id"), m.group("title"), m.start()))

    # Упорядочим по позиции и проставим end
    anchors.sort(key=lambda a: a.start)
    n = len(anchors)
    for i, a in enumerate(anchors):
        a.end = anchors[i + 1].start if i + 1 < n else len(text)

    # Удаляем дубликаты, когда RE_NUM_TITLE пересекается с RE_CLAUSE
    dedup: List[Anchor] = []
    for a in anchors:
        if dedup and abs(a.start - dedup[-1].start) < 2 and a.id == dedup[-1].id:

            # Предпочитаем более "специализированный" тип
            priority = {"section": 1, "appendix": 1, "clause": 2, "numtitle": 3, "table": 9, "figure": 9}
            if priority.get(a.kind, 5) < priority.get(dedup[-1].kind, 5):
                dedup[-1] = a
            continue
        dedup.append(a)

    # Повторно выставим end после дедупликации
    for i, a in enumerate(dedup):
        a.end = dedup[i + 1].start if i + 1 < len(dedup) else len(text)

    return dedup

# ========================================================================
# Аннотация чанков
# ========================================================================

"""
    Возвращает наиболее релевантный якорь для позиции pos:
     -  берем последний якорь, у которого start <= pos;
     -  предпочитаем более низкий уровень (clause/numtitle) над section/appendix
"""

def _best_anchor_for_pos(anchors: List[Anchor], pos: int) -> Optional[Anchor]:
    if not anchors:
        return None

    candidate: Optional[Anchor] = None
    for a in anchors:
        if a.start <= pos:
            candidate = a
        else:
            break
    
    if candidate is None:
        return None

    # Пробуем найти ближайший "более конкретный" якрорь в окрестности
    # Окно: от ближайшего section до текущей позиии
    best = candidate
    for a in reversed(anchors):
        if a.start > pos:
            continue

        # Пропустим слишком общие объекты
        if a.kind in ("table", "figure"):
            continue

        # Выбираем ближайший по позиции и с наибольшей детализацией
        if a.start <= pos and a.start >= best.start:

            # Если новый якорь более детальный
            if (a.level > bext.level and a.level < 99) or (a.level == best.level and a.start >= best.start):
                best = a
    return best

# Для каждого чанка возвращает строковую метку
# Если ничего подходящего не найден - None
def annotate_chunks_with_sections(text: str, chunks: List[str]) -> List[str]:
    anchors = find_anchors(text)
    labels: List[str] = []
    cursor = 0
    for ch in chunks:
        start = current_process
        anchor = _best_anchor_for_pos(anchors, start)
        labels.append(anchor.lebel() if anchor else None)
        cursor += len(ch) + 1
    return labels


# ========================================================================
# Оглавление
# ========================================================================

"""
    Возвращает плоский список якорей (с Level/start/end) - можно использовать для оглавления
    В дальнейшем при необходимости можно собрать иерархическое дерево по level 
"""
def build_outline(text: str) -> List[Anchor]:
    return find_anchors(text)

__all__ = [
    "Anchor",
    "find_anchors",
    "annotate_chunks_with_sections",
    "build_outline",
]