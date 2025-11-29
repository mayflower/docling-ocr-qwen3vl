"""Docling plugin entry point for Qwen3-VL models."""

from __future__ import annotations

from collections.abc import Mapping

from .code_formula_model import Qwen3VlCodeFormulaModel
from .layout_model import Qwen3VlLayoutModel
from .model import Qwen3VlOcrModel
from .picture_classifier_model import Qwen3VlPictureClassifierModel
from .picture_description_model import Qwen3VlPictureDescriptionModel
from .table_structure_model import Qwen3VlTableStructureModel


def ocr_engines() -> Mapping[str, list[type]]:
    """Expose available OCR engine classes to Docling."""
    return {
        "ocr_engines": [
            Qwen3VlOcrModel,
        ]
    }


def picture_description() -> Mapping[str, list[type]]:
    """Expose available picture description model classes to Docling."""
    return {
        "picture_description": [
            Qwen3VlPictureDescriptionModel,
        ]
    }


def table_structure_engines() -> Mapping[str, list[type]]:
    """Expose available table structure model classes to Docling."""
    return {
        "table_structure_engines": [
            Qwen3VlTableStructureModel,
        ]
    }


def layout_engines() -> Mapping[str, list[type]]:
    """Expose available layout model classes to Docling."""
    return {
        "layout_engines": [
            Qwen3VlLayoutModel,
        ]
    }


def picture_classifier() -> Mapping[str, list[type]]:
    """Expose available picture classifier model classes to Docling."""
    return {
        "picture_classifier": [
            Qwen3VlPictureClassifierModel,
        ]
    }


def code_formula() -> Mapping[str, list[type]]:
    """Expose available code/formula model classes to Docling."""
    return {
        "code_formula": [
            Qwen3VlCodeFormulaModel,
        ]
    }
