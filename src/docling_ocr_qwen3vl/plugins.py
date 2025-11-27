"""Docling plugin entry point for Qwen3-VL OCR."""

from __future__ import annotations

from collections.abc import Mapping

from .model import Qwen3VlOcrModel


def ocr_engines() -> Mapping[str, list[type]]:
    """Expose available OCR engine classes to Docling."""
    return {
        "ocr_engines": [
            Qwen3VlOcrModel,
        ]
    }
