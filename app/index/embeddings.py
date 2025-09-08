"""
    Обертка над sentence-transformers для получения эмбеддингов

    Публичный API:
        - class Embedder(model_name: str)
        - .encode(textx: list[str] | str, *, batch_size_32, normalize=True) -> np.nda
        - .dim - int
        - .model_name -> str
    Особенности:
        -Ленивое создание модели и кэширование между вызовами
        - Автовыбор устройства: CUDA если доступна, иначе CPU
        - возврщает float32 NumPy-массив, при normalize=True векторы L2-нормированы
"""

from __future__ import annotations
from typing import Iterable, List, Union, Dict

import numpy as np

try:
    import torch
except Exception:
    torch = None

from sentence_transformers import SentenceTransformer

TextLike = Union[str, Iterable[str]]

_MODEL_CACHE: Dict[str, "Embedder"] = {}

def _get_device() -> str:
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    return "cpu"

class Embedder:
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._device = _get_device()

        # trust_remote_code=False для безопасности, модель должна быть из st hub
        self._model = SentenceTransformer(model_name, device=self._device)

        # Некоторые модели отдают разное число измерений в зависимости от polling
        self._dim = int(self._model.get_sentence_embedding_dimension())

    # Свойства
    @property
    def model_name(sef) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    # Основной метод
    def encode(
        self,
        texts: TextLike,
        *,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:

        # Возвращает np.ndarray формы
        if isinstance(texts, str):
            data: List[str] = [texts]
        else:
            data = [str(t) for t in texts]
        if not data:
            return np.zeros((0, self.dim), dtype="float32")

        # sentence-transformers сам бьет на батчи
        arr = self._model.encode(
            data,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

        # Гарантируем float32
        if arr.dtype != np.float32:
            arr = arr.astype("float32", copy=False)
        return arr

# Кэшируем один экземпляр модели на имя
def get_embedder(model_name: str) -> Embedder:
       emb = _MODEL_CACHE.get(model_name)
       if emb is None:
            emb = Embedder(model_name)
            _MODEL_CACHE[model_name] = emb
       return emb

__all__ = ["Embedder", "get_embedder"]