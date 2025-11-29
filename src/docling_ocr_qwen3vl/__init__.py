"""Docling plugin using Qwen3-VL vision-language model for document processing."""

from ._version import __version__
from .code_formula_model import Qwen3VlCodeFormulaModel
from .layout_model import Qwen3VlLayoutModel
from .model import Qwen3VlOcrModel
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
from .picture_classifier_model import Qwen3VlPictureClassifierModel
from .picture_description_model import Qwen3VlPictureDescriptionModel
from .table_structure_model import Qwen3VlTableStructureModel


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
