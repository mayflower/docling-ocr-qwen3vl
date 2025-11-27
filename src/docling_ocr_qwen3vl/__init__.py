"""Docling OCR plugin using Qwen3-VL vision-language model."""

from ._version import __version__
from .model import Qwen3VlOcrModel
from .options import Qwen3VlOcrOptions, Qwen3VlPromptMode


__all__ = [
    "Qwen3VlOcrModel",
    "Qwen3VlOcrOptions",
    "Qwen3VlPromptMode",
    "__version__",
]
