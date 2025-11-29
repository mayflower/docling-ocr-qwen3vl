"""Prompt templates and helpers for Qwen3-VL OCR."""

from __future__ import annotations

from .options import Qwen3VlPromptMode


# System prompt for QwenVL Document Parser HTML format.
# This is required to trigger proper data-bbox output from Qwen3-VL models.
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/document_parsing.ipynb
QWENVL_HTML_SYSTEM_PROMPT = """You are an AI specialized in recognizing and extracting text from document images.
Your mission is to analyze the page image and generate the result in QwenVL Document Parser HTML format.

QwenVL Document Parser HTML rules:
- Use standard HTML elements (<h1>-<h6>, <p>, <ul>, <ol>, <li>, <table>, <tr>, <th>, <td>, <img>, <figure>, <figcaption>, <div>, etc.).
- Every visible element (text block, heading, table, figure, list item, header, footer) MUST have a data-bbox="x1 y1 x2 y2" attribute with 4 integers in [0, 1000] (page coordinates, top-left origin).
- Coordinates must satisfy: x1 <= x2, y1 <= y2.
- Do NOT output <style>, <script>, <link>, or inline CSS.
- Do NOT output explanations before or after the HTML.
- Output valid HTML that can be parsed with a standard HTML parser.

If you need to reason step-by-step, think inside <think>...</think>.
After </think>, output ONLY the final QwenVL HTML."""


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
    # User prompt for QwenVL HTML - requires QWENVL_HTML_SYSTEM_PROMPT as system message
    Qwen3VlPromptMode.QWENVL_HTML: "QwenVL HTML",
}


def resolve_prompt(mode: Qwen3VlPromptMode, overrides: dict[str, str]) -> str:
    """Return the prompt string for the requested mode."""
    if overrides:
        candidate = overrides.get(mode.value)
        if candidate:
            return candidate
    return DEFAULT_PROMPTS[mode]
