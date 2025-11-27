"""Prompt templates and helpers for Qwen3-VL OCR."""

from __future__ import annotations

from .options import Qwen3VlPromptMode


# Default prompt templates for Qwen3-VL OCR.
# They can be overridden via options.
DEFAULT_PROMPTS: dict[Qwen3VlPromptMode, str] = {
    Qwen3VlPromptMode.OCR: (
        "Extract all text from this image. "
        "Return only the text content, preserving the reading order. "
        "Separate paragraphs with blank lines."
    ),
    Qwen3VlPromptMode.MARKDOWN: (
        "Convert this document image to markdown format. "
        "Preserve headings, lists, tables, and formatting. "
        "Use appropriate markdown syntax for structure."
    ),
    Qwen3VlPromptMode.STRUCTURED: (
        "Extract all text from this document with layout awareness. "
        "Identify and label headings, paragraphs, tables, and lists. "
        "Preserve the document structure and reading order. "
        "Separate distinct sections with blank lines."
    ),
}


def resolve_prompt(mode: Qwen3VlPromptMode, overrides: dict[str, str]) -> str:
    """Return the prompt string for the requested mode."""
    if overrides:
        candidate = overrides.get(mode.value)
        if candidate:
            return candidate
    return DEFAULT_PROMPTS[mode]
