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


# Prompt for table structure extraction
TABLE_STRUCTURE_PROMPT = """Analyze the table in this image and extract its structure.

Output a JSON object with the following format:
{
  "rows": <number of rows>,
  "cols": <number of columns>,
  "cells": [
    {
      "row": <0-indexed row>,
      "col": <0-indexed column>,
      "row_span": <number of rows spanned, default 1>,
      "col_span": <number of columns spanned, default 1>,
      "text": "<cell text content>",
      "is_header": <true if header cell, false otherwise>,
      "bbox": [x1, y1, x2, y2]
    },
    ...
  ]
}

Rules:
- Coordinates are in [0, 1000] scale relative to the table image
- Include ALL cells, including merged cells
- For merged cells, use row_span and col_span > 1
- is_header should be true for header rows/columns
- Output ONLY valid JSON, no explanations"""


# Prompt for layout analysis
LAYOUT_ANALYSIS_PROMPT = """Analyze the document layout in this image.

Identify all document elements and output a JSON array with the following format:
[
  {
    "label": "<element type>",
    "bbox": [x1, y1, x2, y2],
    "confidence": <0.0 to 1.0>,
    "text": "<text content if applicable>"
  },
  ...
]

Element types (use exactly these labels):
- "title" - Main document title
- "section_header" - Section headings
- "text" - Regular paragraph text
- "list_item" - List items (bulleted or numbered)
- "table" - Table regions
- "picture" - Images, figures, diagrams
- "caption" - Figure/table captions
- "footnote" - Footnotes
- "page_header" - Page headers
- "page_footer" - Page footers
- "formula" - Mathematical formulas
- "code" - Code blocks

Rules:
- Coordinates are in [0, 1000] scale relative to the page
- Include ALL visible elements
- Maintain reading order (top to bottom, left to right)
- Output ONLY valid JSON array, no explanations"""


# Prompt for picture classification
PICTURE_CLASSIFICATION_PROMPT = """Classify this image into one or more categories.

Output a JSON object with the following format:
{
  "classes": [
    {"class_name": "<category>", "confidence": <0.0 to 1.0>},
    ...
  ]
}

Available categories (use exactly these names):
- "photograph" - Real-world photographs
- "chart" - Bar charts, line charts, pie charts, etc.
- "diagram" - Technical diagrams, flowcharts, architecture diagrams
- "illustration" - Drawings, sketches, artistic illustrations
- "table" - Tabular data presented as an image
- "map" - Geographic maps
- "screenshot" - Software screenshots
- "logo" - Company logos, brand marks
- "equation" - Mathematical equations or formulas
- "other" - None of the above

Rules:
- Return 1-3 most likely classes
- Confidence values should sum to approximately 1.0
- Output ONLY valid JSON, no explanations"""


# Prompt for code detection
CODE_DETECTION_PROMPT = """Analyze this image and extract any code content.

If this image contains code, output a JSON object:
{
  "is_code": true,
  "language": "<programming language>",
  "code": "<extracted code text>"
}

If this image contains a mathematical formula, output:
{
  "is_formula": true,
  "latex": "<LaTeX representation of the formula>"
}

If neither code nor formula, output:
{
  "is_code": false,
  "is_formula": false
}

Programming languages to detect:
python, javascript, typescript, java, c, cpp, csharp, go, rust, ruby, php, swift, kotlin, sql, bash, html, css, json, yaml, xml, other

Rules:
- Extract the exact code/formula text
- Identify the programming language accurately
- For formulas, provide valid LaTeX
- Output ONLY valid JSON, no explanations"""
