"""
    Хранилище индекса на диске (NumPy + JSONL).

    Состав:
        - embeddings.npy      — матрица эмбеддингов (N, D), dtype=float32, L2-нормирована.
        - meta.jsonl          — метаданные по одному JSON-объекту на строку, индекс соответствует строке в embeddings.npy.
        - cfg.json            — служебная конфигурация индекса (embed_dim).

    Публичный API:
        - ensure() -> None                                   # гарантирует наличие директории/файлов
        - get_embed_dim(default: int | None = None) -> int   # возвращает D (если нет cfg — пишет default, либо ошибка)
        - set_embed_dim(dim: int) -> None                    # записывает D в cfg.json (однократно при создании)
        - load_vecs() -> np.ndarray                          # загружает матрицу эмбеддингов
        - save_vecs(arr: np.ndarray) -> None                 # атомарно сохраняет матрицу эмбеддингов
        - append_meta(items: list[dict]) -> None             # дописывает объекты в meta.jsonl (по одному на строку)
        - load_meta(limit: int | None = None) -> list[dict]  # загружает метаданные (опционально с лимитом)
        - iter_meta() -> Iterator[dict]                      # ленивый итератор по метаданным
        - index_size() -> tuple[int, int]                    # (N, D) из embeddings.npy
        - clear_index() -> None                              # очистка индекса (ОСТОРОЖНО!)

    Примечания:
        - Потокобезопасность: рассчитано на одиночный процесс. Для многопроцессного доступа использовать блокировки.
"""

from __future__ import annotations
from typing import Iterator, List, Tuple, Optional

import json
import os
import pathlib
import tempfile
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "index"
EMB_PATH = INDEX_DIR / "embeddings.npy"
META_PATH = INDEX_DIR / "meta.jsonl"
CFG_PATH = INDEX_DIR / "cfg.json"

# =====================================================================
# Вспомогательные
# =====================================================================

# Атомарная запись: пишем во времменный файл и переименовываем
def _atomic_write_bytes(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent)) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = pathlib.Path(tmp.name)
    tmp_path.replace(path)

def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


# =====================================================================
# Базовые операции
# =====================================================================

# Создать директорию и пустые файлы при отсутсвии
def ensure() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if not EMB_PATH.exists():
        np.save(EMB_PATH, np.zeros((0, 0), dtype="float32"))
    if not META_PATH.exists():
        META_PATH.write_text("", encoding="utf-8")
    if not CFG_PATH.exists():
        CFG_PATH.write_text("{}", encoding="utf-8")

# Прочитать размерность эмбеддинга D из cfg.json. Если отсутсвует - инициализировать default
def get_embed_dim(default: int | None = None) -> int:
    ensure()
    try:
        cfg = json.loads(CFG_PATH.read_text(encoding="utf-8") or "{}")
    except Exception:
        cfg = {}
    dim = cfg.get("embed_dim")
    if dim is None:
        if default is None:
            raise RuntimeError("cfg.json отсутствует поле 'embed_dim'. Установите его через set_embed_dim(dim) "
                               "или передайте default в get_embed_dim().")
        set_embed_dim(int(default))
        return int(default)
    return int(dim)

# Установить размерность эмбеддинга D
def set_embed_dim(dim: int) -> None:
    ensure()
    try:
        cfg = json.loads(CFG_PATH.read_text(encoding="utf-8") or "{}")
    except Exception:
        cfg = {}
    cfg["embed_dim"] = int(dim)
    _atomic_write_text(CFG_PATH, json.dumps(cfg, ensure_ascii=False, indent=2))

# Загрузить матрицу эмбеддингов
def load_vecs() -> np.ndarray:
    ensure()
    arr = np.load(EMB_PATH, allow_pickle=False)
    if arr.ndim != 2:

        # Привести к (0, 0) or (N, D)
        if arr.size == 0:
            return np.zeros((0, 0), dtype="float32")
        raise ValueError(f"embeddings.npy имеет некорректную форму")
    if arr.dtype != np.float32:
        arr = arr.astype("float32", copy=False)
    return arr

# Атомарно сохранить матрицу эмбеддингов
def save_vecs(arr: np.ndarray) -> None:
    ensure()
    if arr.ndim != 2:
        raise ValueError("Ожидается 2D-массив (N, D)")
    if arr.dtype != np.float32:
        arr = arr.astype("float32", copy=False)

    # Инициализировать embed_dim при необходимости
    try:
        current_dim = get_embed_dim(None)
    except RuntimeError:
        curremt_dim = None
    if current_dim is None:
        set_embed_dim(int(arr.shape[1]))
    else:
        if arr.shape[1] != current_dim and arr.shape[0] > 0:
            raise ValueError(f"Несоответствие размерности эмбеддинга: D={arr.shape[1]} != cfg.embed_dim={current_dim}")

    # np.save в буфер и атомарно записать
    with tempfile.TemporaryFile() as tmp:
        np.save(tmp, arr)
        tmp.seek(0)
    _atomic_write_bytes(EMB_PATH, data)

# Добавить ззаписи в meta.jsonl
def append_meta(items: List[dict]) -> None:
    ensure()

    # Записываем батчом одним строковым буфером
    buf = "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    with open(META_PATH, "a", encoding="utf-8") as f:
        f.write(buf)


# Загрузить все (или limit) метадданные. 
def load_meta(limit: Optional[int] = None) -> List[dict]:
    ensure()
    out: List[dict] = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                
                # Пропустим битую строку
                continue
            if limit is not None and len(out) >= limit:
                break
    return out
    
# Ленивый итератор по meta.jsonl
def iter_meta() -> Iterator[dict]:
    ensure()
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

# Вернуть (N, D) - число векторов и размерность
def index_size() -> Tuple[int, int]:
    arr = load_vecs()
    if arr.size == 0:
        try:
            d = get_embed_dim(None)
        except RuntimeError:
            d = 0
        return (0, int(d))
    return (int(arr.shape[0]), int(arr.shape[1]))

# Полная очистка индекса
def clear_index() -> None:
    if EMB_PATH.exists():
        EMB_PATH.unlink()
    if META_PATH.exists():
        META_PATH.unlink()
    if CFG_PATH.exists():
        CFG_PATH.unlink()
    ensure()
