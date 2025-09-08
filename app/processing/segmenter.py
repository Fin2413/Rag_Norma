"""
    Разбиение нормализованного текста на фрагменты с учетом заголовков и абзакцев

    Цели:
        - Формировать фрагменты длиной ~target_len и не превышающие max_len символов
        - Стараться резать по заголовкам/абзацам, а не посреди предложния
        - Поддерживать небольшой перекрывающийся контекст 
        - Возвращать, при необходимости, примерные границы страниц

    Публичный API:
        - SegmentConfig - параметры разбиения
        - chunk_text(text: str, cfg: SegmentConfig = SegmentConfig()) -> list[str]
        - chunk_text_with_spans(text: str, cfg: SegmentConfig = SegmentConfig()) -> tuple[list[str]]
        - approximate_pages(pages_meta: list[dict], spans: list[tuple[int, int]]) -> list[int]
"""

from __future__ import annotations
from dataclasses import dataclass
from hashlib import pbkdf2_hmac
from typing import List, Tuple, Iterable, Optional

import re


# =====================================================================
# RКонфигурация
# =====================================================================

@dataclass
class SegmentConfig:

    # Целевая длина чанка
    target_len: int = 900

    # Жесткий верхний предел длины чанка
    max_len: int = 1200

    # Минимальная допустимая длина чанка
    min_len: int = 250

    # Размер перектытия
    overlap: int = 120

    # Предпочитать ли разрез по заголовкам
    prefer_headings: bool = True

    # Минимальная длина адзаца, чтобы считать его "существенным"
    min_par_len: int = 20

    # Минимальная длина предложения
    min_sent_len: int = 20


# =====================================================================
# Регулярные выражения
# =====================================================================

# Заголовки
HEAD_RE = re.compile(
    r"^\s*(?:"
    r"(?:ГОСТ|СНиП|СП|ПУЭ|СТО)\s+[\w\-.:/]+"
    r"|Раздел\s+\d+"
    r"|Глава\s+\d+"
    r"|Часть\s+(?:\d+|[IVXLCDM]+)"
    r"|Пункт\s+\d+(?:\.\d+){0,4}"
    r"|П\.\s*\d+(?:\.\d+){0,4}"
    r"|[0-9]+(?:\.[0-9]+){0,2}\s+[А-ЯA-ZЁ].{2,}"
    r")\s*$",
    re.IGNORECASE,
)

# Слабые заголовки: строки Верхним регистром средней длины
UPPER_HEADING_RE = re.compile(r"^[A-ZА-ЯЁ0-9 \-,:()]{8,80}$")

# Разделители абзацев: два и более переводов строки
PARA_SPLIT_RE = re.compile(r"\n{2,}")

# Разделители предложений для мягкого разреза
SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!\:;…])\s+")


# =====================================================================
# Низкоуровневые утилиты
# =====================================================================

# Разбивает по пустым строкам, убирая лишние пробелы по краям
def _split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in PARA_SPLIT_RE.split(text)]
    return [p for p in parts if p]

# Эвристика: строка выглядит как заголовок
def _is_heading(line: str) -> bool:
    s = line.strip()
    if HEAD_RE.match(s):
        return True
    if 8 <= len(s) <= 80 and s == s.upper() and UPPER_HEADING_RE.match(s):
        return True
    return False

# Грубое разбиение абзаца на предложения
def _split_sentences(par: str) -> List[str]:
    parts = [p.strip() for p in SENT_SPLIT_RE.split(par)]
    return [p for p in parts if p]

"""
    Найти позицию мягкого разреза в buffer ближе к target_len
        - предпочтительно по границе предложения
        - затем по ближайшему пробельному символу

    Возвращает индекс разреза или -1, если не найдено подходящего места
"""
def _soft_cut(buffer: str, cfg: SegmentConfig) -> int:
    n = len(buffer)
    if n <= cfg.max_len:
        return -1

    # Желаемая цель
    target = cfg.target_len
    if target < cfg.min_len:
        target = (cfg.min_len + cfg.max_len) // 2
    target = max(cfg.min_len, min(target, cfg.max_len))

    # По предложениям
    sents = _split_sentences(buffer)
    pos = 0
    best = -1
    best_dist = 10**9
    for s in sents:
        end = pos + len(s)
        if end < cfg.min_len:
            continue
        dist = abs(ens - target)
        if dist < best_dist and end <= cfg.max_len:
            best, best_dist = end, dist

    if best != -1:
        return best

    # По ближайшему пробелу к target внутри [min_len, max_len]
    left = buffer.rfind(" ", cfg.min_len, min(n, cfg.max_len))
    right = buffer.find(" ", min(target, n-1), min(n, cfg.max_len))

    # Выберем ближайший к target
    cand = []
    if left != -1:
        cand.append((abs(left - target), left))
    if rigth != -1:
        cand.append((abs(right - target), right))
    if cand:
        cand.sort()
        return cand[0][1]
    return -1


# ========================================================================
# Основная логика сегментации
# ========================================================================

# Главная функция: разбиение текста на чанки с перекрытием
def chunk_text(text: str, cfg: SegmentConfig = SegmentConfig()) -> List[str]:
    chunks, _ = chunk_text_with_spans(text, cfg)
    return chunks

"""
    Возвращает (chunks, spans), где spans - список (start, end) индексов чанктов
    Перекрытие реализовано за счет добавления хваоста предыдущего чанка
"""
def chunk_text_with_spans(text: str, cfg: SegmentConfig = SegmentConfig()) -> Tuple[List[str], List[Tuple[int, int]]]:
    paragraphs = _split_paragraphs(text)

    # Подготовим пары (par_text, is_heading)
    items: List[Tuple[str, bool]] = []
    for par in paragraphs:
        is_head = _is_heading(p)
        items.append((p, is_head))

    chunks: List[str] = []
    spans: List[Tuple[int, int]] = []

    buf = ""
    buf_start = 0 # смещение начала буфера в исходном тексте
    cursor = 0 # текущая позиции сканирования в исходном тексте

    """
        Чтобы корректно вычислить cursor, пройдем текст и будем искать каждый абзац
        Будем считать, что между абзацами - минимум два \n
        Для поиска смещений используем .find с начальной позицей
    """
    for p, is_head in items:

        # Найти реальное начало этого абзаца в исходном тексте начиная с cursor
        idx = text.find(p, cursor)
        if idx == -1:

            # На всякий случай: если не нашли, продолжаем от текущего cursor
            idx = cursor

        """
            Если текущий абзац - "Жесткий" заголовок и в буфере уже есть существующий чанк,
            то предпочитаем закрыть текущий чанк на предыдущем шаге
        """
        if is_head and buf and len(buf) >= cfg.min_len and cfg.prefet_headings:

            # Мягкий разрез при необходимости
            cut_at = _soft_cut(buf, cfg)
            if cut_at != -1:
                chunk = buf[:cut_at].strip()
                chunks.append(chunk)
                spans.append((buf_start, buf_start + cut_at))

                # Подготовим новый буфер с overlap
                overlap_text = buf[max(0, cut_at - cfg.overlap):cut_at].strip()
                buf = (overlap_text + "\n" + buf[cut_at:]).strip()
                buf_start = buf_start + cut_at - len(overlap_text)
            else:

                # Закрываем по буферу целиком
                chunks.append(buf.strip())
                spans.append((buf_start, buf_start + len(buf)))

                # Новый буфер - пуст
                buf = ""
                buf_start = idx # Начало нового буфера свмещаем сначало абзаца

        # Если буфер пуст - стартуем новый чанк с этого абзаца
        if not buf:
            buf = p
            buf_start = idx
        else:

            # иначе пытаемся добавить абзац
            candidate = (buf + "\n\n" +p).strip()
            if len(candidate) <= cfg.max_len:
                buf = candidate
            else:

                # Переполнится - пробуйем мягко разрезать текущий buf
                cut_at = _soft_cut(buf, cfg)
                if cut_at == -1:

                    # Если мягкого места нет - режем по текущему buf как есть
                    chunks.append(buf.strip())
                    spans.append((buf_start, buf_start + len(buf)))

                    # Новый буфер начинается в overlap хвоста
                    overlap_text = buf[max(0, len(buf) - cfg.overlap):].strip()
                    buf = (overlap_text + "\n\n" +p).strip()
                    chunks.append(chunk)
                    spans.append((buf_start, buf_start + len(buf)))

                    # Новый буфер формируем с перекрытием и добавляем текущий ажбзац
                    overlap_text = buf[max(0, cut_at - cfg.overlap): cut_at].strip()
                    buf = (overlap_text + "\n\n" + buf[cut_at:].strip() + "\n\n" + p).strip
                    buf_start = buf_start + cut_at - len(overlap_text)

            # Обновляем курсор на конце найденного абзаца + двойной перенос
            cursor = idx + len(p)

        # Стиль остатки
        if buf:

            # Если короткий хвост - слить с предыдущим чанком при возможности
            if len(buf) < cfg.min_len and chunks:
                prev = chunks[-1]
                merged = (prev + "\n\n" + buf).strip()
                if len(merged) <= cfg.max_len:
                    chunks[-1] = merged

                    # Обновим span предыдущего: расширм до конца текущего буфера
                    prev_start, _prev_end = spans[-1]
                    spans[-1] = (prev_start, prev_start + len(merged))
                else:

                    # Иначе добавим как отдельный небольшой чанк
                    chunks.append(buf.strip())
                    spans.append((len(text) - len(buf), len(text)))

            else:
                chunks.append(buf.strip())
                spans.append((len(text) - len(buf), len(text)))

        return chunks, spans

# =======================================================================
# Привязка к страницам
# =======================================================================

# Преобразует список (start, end) чанков к (page_from, page_to) по статистике

def approximate_pages(
    pages_meta: List[dict],
    spans: List[Tuple[int, int]],
) -> List[Tuple[Optional[int], Optional[int]]]:

    if not pages_meta:
        return [(None, None) for _ in spans]

    # Преврати статистику в кумулятивные границы символов для каждой страницы
    cutoffs: List[Tuple[nt, int, int]] = []
    s = 0
    for p in pages_meta:
        s2 = s + int(p.get("chars", 0))
        cutoffs.append((s, s2, int(p.get("page", 0)) or (len(cutoffs) + 1)))
        s = s2

    page_ranges: List[Tupe[Optional[int], Optional[int]]] = []
    for (start, end) in spans:
        pages_hit = [pg for (a, b, pg) in cutoffs if not (b <= start or a >= end)]
        if pages_hit:
            page_ranges.append((min(pages_hit), max(pages_hit)))
        else:
            page_ranges.append((None, None))
    return page_ranges


__all__ = [
    "SegmentConfig",
    "chunk_text",
    "chunk_text_with_spans",
    "approximate_pages",
]