"""Qwen3-VL Layout Analysis Model for Docling."""

from __future__ import annotations

import json
import logging
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

from ._model_registry import get_model, maybe_empty_cache
from ._vlm_jsonformer import generate_json_single_shot
from .options import Qwen3VlLayoutOptions
from .prompts import LAYOUT_ANALYSIS_PROMPT


_log = logging.getLogger(__name__)


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
        """Analyze page layout using Qwen3-VL with assistant-prefix generation."""
        assert self._shared is not None
        model = self._shared.model
        processor = self._shared.processor

        image_rgb = page_image.convert("RGB")

        elements = generate_json_single_shot(
            model=model,
            processor=processor,
            prompt=LAYOUT_ANALYSIS_PROMPT,
            image=image_rgb,
            max_new_tokens=self.options.max_new_tokens,
            root_type="array",
        )
        maybe_empty_cache()

        _log.debug(
            "Layout output (page %s, %d elements): %s",
            page.page_no,
            len(elements),
            json.dumps(elements)[:500],
        )

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

            # Validate and fix coordinates
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            if x1 == x2 or y1 == y2:
                _log.debug("Skipping zero-area element: %s", elem)
                continue

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
