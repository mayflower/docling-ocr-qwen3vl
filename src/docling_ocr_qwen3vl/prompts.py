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

Output ONLY the final QwenVL HTML, no explanations."""


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


# Prompt for table structure extraction — flat coordinates to avoid
# nested-array formatting errors from small VLMs.
TABLE_STRUCTURE_PROMPT = """Extract the table structure from this image.

Return a JSON object:
{"rows":<int>,"cols":<int>,"cells":[{"row":<int>,"col":<int>,"text":"<content>","rs":<row_span>,"cs":<col_span>,"hdr":<true/false>,"x1":<int>,"y1":<int>,"x2":<int>,"y2":<int>},...]}

Coordinates are integers 0-1000 relative to the table image.
Row/col are 0-indexed. rs/cs default to 1. hdr is true for header cells.

Output ONLY the JSON object."""


# Prompt for layout analysis — uses flat coordinate fields to avoid
# nested-array formatting errors from small VLMs.
LAYOUT_ANALYSIS_PROMPT = """Detect every document element in this page image.

Return a JSON array. Each element:
{"label":"<type>","x1":<int>,"y1":<int>,"x2":<int>,"y2":<int>}

Rules:
- Coordinates are integers 0-1000 (top-left origin)
- x1 < x2, y1 < y2 (y1 is top edge, y2 is bottom edge)
- Use the correct label for each element

Types: title, section_header, text, list_item, table, picture, caption, footnote, page_header, page_footer, formula, code

Example for a page with a header, title, paragraph, and table:
[{"label":"page_header","x1":50,"y1":10,"x2":950,"y2":40},{"label":"title","x1":100,"y1":50,"x2":800,"y2":100},{"label":"text","x1":50,"y1":120,"x2":950,"y2":450},{"label":"table","x1":50,"y1":470,"x2":950,"y2":750}]

Output ONLY the JSON array."""


# Shorter prompts for constrained (jsonformer) generation — the schema is
# injected by VLMJsonformer automatically, so we only need the task description.
LAYOUT_JSONFORMER_PROMPT = (
    "Detect every document element in this page image. "
    "Types: title, section_header, text, list_item, table, picture, "
    "caption, footnote, page_header, page_footer, formula, code. "
    "Coordinates are integers 0-1000 (top-left origin)."
)

TABLE_JSONFORMER_PROMPT = (
    "Extract the table structure from this image. "
    "Row/col are 0-indexed. rs/cs default to 1. hdr is true for header cells. "
    "Coordinates are integers 0-1000 relative to the table image."
)


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
