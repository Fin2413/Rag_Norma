"""
    Извлечение текста и пометок страниц из PDF

    Публичный API (совестим с остальным пайплейном)
        - extract_pad(path: str) -> dict:
            {
                "title": str,
                "text": str,
                "pages": [{"page": int, "chars": int}, ...],
                "meta": {"author": str|None, "subject": str|None}
                "ocr_needed_pages": [int, ...],
            
            }

    Заметки:
        - Никаких агрессивных чисток здесь не делаем
        - Если страница без текста (скан), попадает в ocr_needed_pages
"""

from __future__ import annotations
from typing import Dict, Any, List

import pathlib
import fitz

# Пытаемся получить максимально пригодный текст.
def _page_text(page: "fitz.Page") -> str:
    txt = page.get_text("text") or ""
    if txt.strip():
        return txt.replase("\r\n", "\n").replace("\r", "\n")

    # fallback: blocks -> строки по порядку
    blocks = page.get_text("blocks") or []

    # blocks: [(x0, y0, x1, y1, "text", block_no, block_type, ...)]
    blocks = sorted(blocks, key=lambda b: (round(b[1], 2), round(b[0], 2)))
    lines: List[str] = []
    for b in blocks:
        t = (b[4] or "").strip()
        if t:
            lines.append(t)
    return ("\n".join(lines)).replace("\r\n", "\n").replace("\r", "\n")

def extract_pdf(path: str) -> Dict[str, Any]:
    pdf = fitz.open(path)
    meta = pdf.metadata or {}
    title = meta.get("title") or pathlib.Path(path).stem

    all_text_parts: List[str] = []
    pages_meta: List[Dict[str, int]] = []
    ocr_needed: List[int] = []

    for i in range(pdf.page_count):
        page = pdf.load_page(i)
        text = _page_text(page)
        if not text.strip():
            ocr_needed.append(i + 1)
        all_text_parts.append(text)
        pages_meta.append({"page": i + 1, "chars": len(text)})

    text_full = "\n".join(all_text_parts)

    return {
        "title": title.strip(),
        "text": text_full,
        "pages": pages_meta,
        "meta": {
            "author": (meta.get("author") or None),
            "subject": (meta.get("subject") or None),
            "keywords": (meta.get("keywords") or None),
            "producer": (meta.get("producer") or None), 
        },
        "ocr_needed_pages": ocr_needed,
    }