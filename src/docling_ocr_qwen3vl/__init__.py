"""Docling plugin using Qwen3-VL vision-language model for document processing."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from ._version import __version__
from .options import (
    Qwen3VlCodeFormulaOptions,
    Qwen3VlLayoutOptions,
    Qwen3VlOcrOptions,
    Qwen3VlPictureClassifierOptions,
    Qwen3VlPictureDescriptionOptions,
    Qwen3VlPromptMode,
    Qwen3VlQuantization,
    Qwen3VlTableStructureOptions,
)

if TYPE_CHECKING:
    from .code_formula_model import Qwen3VlCodeFormulaModel
    from .layout_model import Qwen3VlLayoutModel
    from .model import Qwen3VlOcrModel
    from .picture_classifier_model import Qwen3VlPictureClassifierModel
    from .picture_description_model import Qwen3VlPictureDescriptionModel
    from .table_structure_model import Qwen3VlTableStructureModel


_MODEL_IMPORTS = {
    "Qwen3VlOcrModel": (".model", "Qwen3VlOcrModel"),
    "Qwen3VlPictureDescriptionModel": (
        ".picture_description_model",
        "Qwen3VlPictureDescriptionModel",
    ),
    "Qwen3VlTableStructureModel": (
        ".table_structure_model",
        "Qwen3VlTableStructureModel",
    ),
    "Qwen3VlLayoutModel": (".layout_model", "Qwen3VlLayoutModel"),
    "Qwen3VlPictureClassifierModel": (
        ".picture_classifier_model",
        "Qwen3VlPictureClassifierModel",
    ),
    "Qwen3VlCodeFormulaModel": (".code_formula_model", "Qwen3VlCodeFormulaModel"),
}


def __getattr__(name: str) -> Any:
    if name in _MODEL_IMPORTS:
        module_name, attr_name = _MODEL_IMPORTS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Models
    "Qwen3VlOcrModel",
    "Qwen3VlPictureDescriptionModel",
    "Qwen3VlTableStructureModel",
    "Qwen3VlLayoutModel",
    "Qwen3VlPictureClassifierModel",
    "Qwen3VlCodeFormulaModel",
    # Options
    "Qwen3VlOcrOptions",
    "Qwen3VlPictureDescriptionOptions",
    "Qwen3VlTableStructureOptions",
    "Qwen3VlLayoutOptions",
    "Qwen3VlPictureClassifierOptions",
    "Qwen3VlCodeFormulaOptions",
    # Enums
    "Qwen3VlPromptMode",
    "Qwen3VlQuantization",
    # Version
    "__version__",
]
