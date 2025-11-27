import math

from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_ocr_qwen3vl.model import Qwen3VlOcrModel
from docling_ocr_qwen3vl.options import Qwen3VlOcrOptions, Qwen3VlPromptMode
from docling_ocr_qwen3vl.plugins import ocr_engines
from docling_ocr_qwen3vl.prompts import resolve_prompt
from docling_ocr_qwen3vl.qwen_runner import _select_attention_backend, _split_paragraphs


def test_paragraph_splitting_with_blank_lines():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    paragraphs = _split_paragraphs(text)
    assert len(paragraphs) == 3
    assert paragraphs[0] == "First paragraph."
    assert paragraphs[1] == "Second paragraph."
    assert paragraphs[2] == "Third paragraph."


def test_paragraph_splitting_single_newlines_fallback():
    text = "Line one.\nLine two.\nLine three."
    paragraphs = _split_paragraphs(text)
    assert len(paragraphs) == 3
    assert paragraphs[0] == "Line one."
    assert paragraphs[1] == "Line two."
    assert paragraphs[2] == "Line three."


def test_paragraph_splitting_empty_text():
    assert _split_paragraphs("") == []
    assert _split_paragraphs("   ") == []


def test_paragraph_splitting_single_paragraph():
    text = "Just one paragraph with no breaks."
    paragraphs = _split_paragraphs(text)
    assert len(paragraphs) == 1
    assert paragraphs[0] == "Just one paragraph with no breaks."


def test_paragraphs_to_cells_distribution():
    paragraphs = ["First", "Second", "Third"]
    rect = BoundingBox(
        l=0,
        t=0,
        r=200,
        b=300,
        coord_origin=CoordOrigin.TOPLEFT,
    )
    cells = Qwen3VlOcrModel._paragraphs_to_cells(
        paragraphs,
        rect=rect,
        index_offset=0,
    )

    assert len(cells) == 3

    # Check that cells are distributed evenly
    first, second, third = cells
    assert first.index == 0
    assert second.index == 1
    assert third.index == 2

    # Check vertical distribution (height 300 / 3 paragraphs = 100 each)
    first_box = first.rect.to_bounding_box()
    assert math.isclose(first_box.t, 0.0, abs_tol=1e-6)
    assert math.isclose(first_box.b, 100.0, abs_tol=1e-6)

    second_box = second.rect.to_bounding_box()
    assert math.isclose(second_box.t, 100.0, abs_tol=1e-6)
    assert math.isclose(second_box.b, 200.0, abs_tol=1e-6)

    third_box = third.rect.to_bounding_box()
    assert math.isclose(third_box.t, 200.0, abs_tol=1e-6)
    assert math.isclose(third_box.b, 300.0, abs_tol=1e-6)


def test_paragraphs_to_cells_with_offset():
    paragraphs = ["Test"]
    rect = BoundingBox(l=0, t=0, r=100, b=100, coord_origin=CoordOrigin.TOPLEFT)
    cells = Qwen3VlOcrModel._paragraphs_to_cells(
        paragraphs,
        rect=rect,
        index_offset=5,
    )
    assert len(cells) == 1
    assert cells[0].index == 5


def test_plugin_registration_lists_qwen3vl_model():
    registered = ocr_engines()
    assert "ocr_engines" in registered
    assert Qwen3VlOcrModel in registered["ocr_engines"]


def test_default_prompts_contain_expected_content():
    ocr_prompt = resolve_prompt(Qwen3VlPromptMode.OCR, {})
    markdown_prompt = resolve_prompt(Qwen3VlPromptMode.MARKDOWN, {})
    structured_prompt = resolve_prompt(Qwen3VlPromptMode.STRUCTURED, {})

    assert "Extract all text" in ocr_prompt
    assert "reading order" in ocr_prompt

    assert "markdown" in markdown_prompt.lower()
    assert "headings" in markdown_prompt.lower()

    assert "layout" in structured_prompt.lower()


def test_prompt_overrides():
    custom_prompt = "My custom OCR prompt"
    overrides = {Qwen3VlPromptMode.OCR.value: custom_prompt}
    result = resolve_prompt(Qwen3VlPromptMode.OCR, overrides)
    assert result == custom_prompt


def test_attention_backend_falls_back_when_flash_attn_missing(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "flash_attn", None)
    assert _select_attention_backend("flash_attention_2") == "eager"
    assert _select_attention_backend("eager") == "eager"


def test_options_defaults():
    options = Qwen3VlOcrOptions()
    assert options.kind == "qwen3vl_ocr"
    assert options.model_repo_id == "Qwen/Qwen3-VL-8B-Thinking"
    assert options.device == "cuda"
    assert options.max_new_tokens == 4096
    assert options.temperature == 0.6
    assert options.top_p == 0.95
    assert options.top_k == 20
