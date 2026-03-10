"""Runtime wrapper around the Qwen3-VL model for OCR."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from PIL import Image

from ._model_registry import SharedModel, extract_after_think_token, get_model, maybe_empty_cache
from .options import Qwen3VlOcrOptions, Qwen3VlPromptMode
from .prompts import QWENVL_HTML_SYSTEM_PROMPT, resolve_prompt


_log = logging.getLogger(__name__)


@dataclass(slots=True)
class HtmlElement:
    """A parsed HTML element with optional bounding box from QwenVL HTML output."""

    tag: str
    text: str
    bbox: tuple[int, int, int, int] | None = None  # (x1, y1, x2, y2)
    element_type: str = "text"  # heading, paragraph, table, figure, list, etc.


@dataclass(slots=True)
class Qwen3VlResult:
    """Container for Qwen3-VL inference output."""

    text: str
    paragraphs: list[str]
    html_elements: list[HtmlElement] = field(default_factory=list)
    raw_html: str | None = None


class Qwen3VlRunner:
    """Lazy-load and execute the Qwen3-VL model for OCR."""

    def __init__(self, options: Qwen3VlOcrOptions):
        self.options = options
        self._shared: SharedModel | None = None

    def ensure_loaded(self) -> None:
        """Load the underlying model if not already available."""
        if self._shared is not None:
            return

        self._shared = get_model(
            model_repo_id=self.options.model_repo_id,
            device=self.options.device,
            dtype=self.options.dtype,
            trust_remote_code=self.options.trust_remote_code,
            hf_token=self.options.hf_token,
            attn_implementation=self.options.attn_implementation,
            quantization=self.options.quantization,
            bnb_4bit_quant_type=self.options.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.options.bnb_4bit_use_double_quant,
        )

    def run(self, image: Image.Image, *, prompt_mode: Qwen3VlPromptMode) -> Qwen3VlResult:
        """Execute inference for a single image."""
        self.ensure_loaded()

        assert self._shared is not None
        model = self._shared.model
        processor = self._shared.processor

        prompt = resolve_prompt(prompt_mode, self.options.prompt_overrides)
        image_rgb = image.convert("RGB")

        # Build chat messages for Qwen3-VL
        # For QWENVL_HTML mode, use system prompt to trigger proper data-bbox output
        if prompt_mode == Qwen3VlPromptMode.QWENVL_HTML:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": QWENVL_HTML_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_rgb},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_rgb},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

        # Process inputs using chat template
        text_input = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[text_input],
            images=[image_rgb],
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to model device
        inputs = inputs.to(model.device)

        _log.debug(
            "Running Qwen3-VL OCR with prompt mode `%s` (max_new_tokens=%s)",
            prompt_mode.value,
            self.options.max_new_tokens,
        )

        # Generate with Qwen3-VL
        # For QWENVL_HTML mode, use deterministic decoding for stable structured output
        import torch

        if prompt_mode == Qwen3VlPromptMode.QWENVL_HTML:
            # Deterministic decoding for structured HTML output
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.options.max_new_tokens,
                    do_sample=False,
                )
        else:
            # Standard sampling for other modes
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.options.max_new_tokens,
                    temperature=self.options.temperature if self.options.do_sample else None,
                    top_p=self.options.top_p if self.options.do_sample else None,
                    top_k=self.options.top_k if self.options.do_sample else None,
                    do_sample=self.options.do_sample,
                )

        # Decode only the generated tokens (exclude input)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, input_len:]

        # For Thinking models: extract content after </think> token (ID 151668)
        # This must be done BEFORE decoding since skip_special_tokens strips the markers
        generated_ids = extract_after_think_token(generated_ids)

        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        output_text = output_text.strip()

        # Handle QWENVL_HTML mode specially - parse HTML with bounding boxes
        if prompt_mode == Qwen3VlPromptMode.QWENVL_HTML:
            _log.debug("QWENVL_HTML raw output: %s", output_text[:500])
            html_elements = _parse_qwenvl_html(output_text)
            _log.debug(
                "QWENVL_HTML parsed %d elements, %d with bbox",
                len(html_elements),
                sum(1 for e in html_elements if e.bbox is not None),
            )
            # Extract plain text from HTML elements for compatibility
            plain_text = "\n\n".join(el.text for el in html_elements if el.text.strip())
            paragraphs = [el.text for el in html_elements if el.text.strip()]

            maybe_empty_cache()

            return Qwen3VlResult(
                text=plain_text,
                paragraphs=paragraphs,
                html_elements=html_elements,
                raw_html=output_text,
            )

        paragraphs = _split_paragraphs(output_text)

        maybe_empty_cache()

        return Qwen3VlResult(text=output_text, paragraphs=paragraphs)


def _parse_qwenvl_html(html_text: str) -> list[HtmlElement]:
    """Parse QwenVL HTML output and extract elements with bounding boxes.

    QwenVL HTML format uses data-bbox attributes with coordinates like:
    <h1 data-bbox="879 283 1605 348">Title</h1>
    <p data-bbox="100 200 500 250">Paragraph text</p>
    """
    from html.parser import HTMLParser

    # Map HTML tags to element types
    tag_to_type = {
        "h1": "heading",
        "h2": "heading",
        "h3": "heading",
        "h4": "heading",
        "h5": "heading",
        "h6": "heading",
        "p": "paragraph",
        "table": "table",
        "tr": "table_row",
        "td": "table_cell",
        "th": "table_cell",
        "ul": "list",
        "ol": "list",
        "li": "list_item",
        "img": "figure",
        "figure": "figure",
        "div": "block",
        "span": "text",
    }

    elements: list[HtmlElement] = []

    class QwenVLHTMLParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.current_tag: str | None = None
            self.current_bbox: tuple[int, int, int, int] | None = None
            self.current_text: list[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            tag = tag.lower()
            if tag in tag_to_type:
                self.current_tag = tag
                self.current_text = []
                self.current_bbox = None

                # Look for data-bbox attribute
                for attr_name, attr_value in attrs:
                    if attr_name == "data-bbox" and attr_value:
                        parts = attr_value.split()
                        if len(parts) == 4:
                            try:
                                self.current_bbox = (
                                    int(parts[0]),
                                    int(parts[1]),
                                    int(parts[2]),
                                    int(parts[3]),
                                )
                            except ValueError:
                                pass

        def handle_endtag(self, tag: str) -> None:
            tag = tag.lower()
            if tag == self.current_tag and self.current_tag in tag_to_type:
                text = " ".join(self.current_text).strip()
                text = re.sub(r"\s+", " ", text)
                if text:
                    elements.append(
                        HtmlElement(
                            tag=self.current_tag,
                            text=text,
                            bbox=self.current_bbox,
                            element_type=tag_to_type[self.current_tag],
                        )
                    )
                self.current_tag = None
                self.current_bbox = None
                self.current_text = []

        def handle_data(self, data: str) -> None:
            if self.current_tag:
                self.current_text.append(data)

    parser = QwenVLHTMLParser()
    try:
        parser.feed(html_text)
    except Exception:
        pass

    # If no structured elements found, fall back to plain text extraction
    if not elements and html_text.strip():
        plain_text = re.sub(r"<[^>]+>", " ", html_text).strip()
        plain_text = re.sub(r"\s+", " ", plain_text)
        if plain_text:
            elements.append(
                HtmlElement(tag="p", text=plain_text, bbox=None, element_type="paragraph")
            )

    return elements


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs based on blank lines."""
    if not text:
        return []

    # Split on double newlines or more

    parts = re.split(r"\n\s*\n", text)
    paragraphs = []
    for part in parts:
        cleaned = part.strip()
        if cleaned:
            paragraphs.append(cleaned)

    # If no paragraph breaks found, split on single newlines as fallback
    if len(paragraphs) <= 1 and text.strip():
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) > 1:
            paragraphs = lines

    return paragraphs if paragraphs else [text.strip()] if text.strip() else []
