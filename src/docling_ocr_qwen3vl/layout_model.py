"""Qwen3-VL Layout Analysis Model for Docling."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import BoundingBox, Cluster, LayoutPrediction, Page
from docling.datamodel.document import ConversionResult
from docling.models.base_layout_model import BaseLayoutModel
from docling_core.types.doc import DocItemLabel

try:
    from docling.datamodel.pipeline_options import BaseLayoutOptions
except ImportError:
    from docling.datamodel.pipeline_options import LayoutOptions as BaseLayoutOptions

from ._model_registry import extract_after_think_token, get_model, maybe_empty_cache
from .options import Qwen3VlLayoutOptions
from .prompts import LAYOUT_ANALYSIS_PROMPT


_log = logging.getLogger(__name__)


def _repair_json(text: str) -> str:
    """Try to fix common JSON issues produced by small VLMs."""
    s = text.strip()
    # Remove trailing commas before ] or }
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # Ensure the array is closed
    if s.startswith("[") and not s.endswith("]"):
        # Find the last complete object and close the array
        last_brace = s.rfind("}")
        if last_brace > 0:
            s = s[: last_brace + 1] + "]"
    return s


def _parse_layout_json(output_text: str, page_no: int) -> list[dict]:
    """Parse layout JSON from model output with repair fallback."""
    json_match = re.search(r"\[[\s\S]*\]", output_text)
    if not json_match:
        # Try to find a partial array and close it
        partial = re.search(r"\[[\s\S]*", output_text)
        if partial:
            repaired = _repair_json(partial.group())
            try:
                elements = json.loads(repaired)
                _log.info(
                    "Layout JSON repaired for page %s (%d elements)",
                    page_no,
                    len(elements),
                )
                return elements
            except json.JSONDecodeError:
                pass
        _log.warning(
            "No JSON array found in layout output (page %s): %s",
            page_no,
            output_text[:300],
        )
        return []

    raw = json_match.group()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        _log.debug("Layout JSON parse failed (page %s), attempting repair: %s", page_no, e)
        repaired = _repair_json(raw)
        try:
            elements = json.loads(repaired)
            _log.info(
                "Layout JSON repaired for page %s (%d elements)",
                page_no,
                len(elements),
            )
            return elements
        except json.JSONDecodeError as e2:
            _log.warning(
                "Failed to parse layout JSON (page %s): %s\nRaw: %s",
                page_no,
                e2,
                output_text[:300],
            )
            return []


# Map from prompt labels to DocItemLabel
LABEL_MAP = {
    "title": DocItemLabel.TITLE,
    "section_header": DocItemLabel.SECTION_HEADER,
    "text": DocItemLabel.TEXT,
    "paragraph": DocItemLabel.TEXT,
    "list_item": DocItemLabel.LIST_ITEM,
    "table": DocItemLabel.TABLE,
    "picture": DocItemLabel.PICTURE,
    "figure": DocItemLabel.PICTURE,
    "caption": DocItemLabel.CAPTION,
    "footnote": DocItemLabel.FOOTNOTE,
    "page_header": DocItemLabel.PAGE_HEADER,
    "page_footer": DocItemLabel.PAGE_FOOTER,
    "formula": DocItemLabel.FORMULA,
    "code": DocItemLabel.CODE,
}


class Qwen3VlLayoutModel(BaseLayoutModel):
    """Qwen3-VL based layout analysis model.

    Uses the Qwen3-VL vision-language model to detect document layout
    including text blocks, headings, tables, figures, etc.
    """

    def __init__(
        self,
        artifacts_path: Path | None,
        accelerator_options: AcceleratorOptions,
        options: Qwen3VlLayoutOptions,
        enabled: bool = True,
        **kwargs,
    ):
        self.enabled = enabled
        self.options = options
        self._shared = None

        if self.enabled:
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

    @classmethod
    def get_options_type(cls) -> type[BaseLayoutOptions]:
        return Qwen3VlLayoutOptions

    def predict_layout(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[LayoutPrediction]:
        """Produce layout predictions for the provided pages."""
        predictions = []

        for page in pages:
            page_image = page.get_image(scale=2.0)
            if page_image is None:
                predictions.append(LayoutPrediction(clusters=[]))
                continue

            clusters = self._analyze_layout(page_image, page)
            predictions.append(LayoutPrediction(clusters=clusters))

        return predictions

    def _analyze_layout(self, page_image, page: Page) -> list[Cluster]:
        """Analyze page layout using Qwen3-VL."""
        import torch

        assert self._shared is not None
        model = self._shared.model
        processor = self._shared.processor

        image_rgb = page_image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_rgb},
                    {"type": "text", "text": LAYOUT_ANALYSIS_PROMPT},
                ],
            }
        ]

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
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=self.options.max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, input_len:]
        generated_ids = extract_after_think_token(generated_ids)

        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        maybe_empty_cache()

        _log.debug("Layout model raw output (page %s): %s", page.page_no, output_text[:500])

        elements = _parse_layout_json(output_text, page.page_no)

        # Convert to Cluster objects
        return self._build_clusters(elements, page)

    def _build_clusters(self, elements: list[dict], page: Page) -> list[Cluster]:
        """Build Cluster objects from parsed JSON elements."""
        clusters = []

        for idx, elem in enumerate(elements):
            label_str = elem.get("label", "text").lower()
            confidence = elem.get("confidence", 0.9)

            # Map label string to DocItemLabel
            label = LABEL_MAP.get(label_str, DocItemLabel.TEXT)

            # Support both flat (x1,y1,x2,y2) and nested (bbox) coordinate formats
            bbox_data = elem.get("bbox")
            if bbox_data and len(bbox_data) == 4:
                x1, y1, x2, y2 = bbox_data
            else:
                x1 = elem.get("x1", 0)
                y1 = elem.get("y1", 0)
                x2 = elem.get("x2", 1000)
                y2 = elem.get("y2", 1000)

            # Convert bbox from 0-1000 scale to page coordinates
            if page.size:
                bbox = BoundingBox(
                    l=(x1 / 1000) * page.size.width,
                    t=(y1 / 1000) * page.size.height,
                    r=(x2 / 1000) * page.size.width,
                    b=(y2 / 1000) * page.size.height,
                )
            else:
                bbox = BoundingBox(l=0, t=0, r=100, b=100)

            clusters.append(
                Cluster(
                    id=idx,
                    label=label,
                    bbox=bbox,
                    confidence=confidence,
                )
            )

        return clusters
