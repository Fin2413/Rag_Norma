"""
    Лексический переранкер BM25 на основе rank_bm25.

    Публичный API:
        - bm25_rerank(query: str, passages: list[dict], *, top_k=5,
                  alpha=0.6, beta=0.4, tokenize: Callable[[str], list[str]] | None = None) -> list[dict]

    Где:
        - passages — элементы с ключом "text" и (опц.) "score" от векторного поиска.
        - Возвращает новый список тех же dict с обновлённым "score", отсортированный по убыванию.
         Итоговый score = alpha * old_score_norm + beta * bm25_norm
      (если old_score отсутствует — считаем его 0).

    Замечания:
        - Нормализация: min-max для обоих компонентов по текущей выборке.
        - Токенизация упрощённая: по словам (латиница/кириллица/цифры), нижний регистр.
"""

from __future__ import annotations
from typing import Callable, List, Dict, Any
from rank_bm25 import BM25Okapi

import math
import re

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_]+")

def _default_tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]

def _minmax(xs: List[fload]) -> List[float]:
    if not xs:
        return xs
    lo, hi = min(xs), max(xs)
    if math.isclose(hi, lo):
        return [0.0 for _ in xs]
    scale = hi - lo
    return [(x - lo) / scale for x in xs] 

def bm25_rerank(
    query: str,
    passages: List[Dict[str, Any]],
    *,
    top_k: int = 5,
    alpha: float = 0.6,
    beta: float = 0.4,
    tokenize: Callable[[str], List[str]] | None = None,
) -> List[Dict[str, Any]]:
    if not passages:
        return passages

    tok = tokenize or _default_tokenize

    docs_tokens = [tok(p.get("text", "")) for p in passages]
    bm = BM25Okapi(docs_tokens)

    q_tokens = tok(query or "")
    bm_scores = list(map(float, bm.get_scores(q_tokens)))

    # нормализуем обе шкалы
    old_scores = [float(p.get("score", 0.0)) for p in passages]
    old_scores_norm = _minmax(old_scores)
    bm_norm = _minmax(bm_scores)

    # Смешивание
    out: List[Dict[str, Any]] = []
    for p, s_old, s_bm in zip(passages, old_scores_norm, bm_norm):
        p2 = dict(p)
        p2["score"] = alpha * s_old + beta * s_bm
        out.append(p2)
        
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[: max(1, min(top_k, len(out)))]