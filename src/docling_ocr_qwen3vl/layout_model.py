"""Qwen3-VL Layout Analysis Model for Docling."""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Sequence
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import BoundingBox, Cluster, LayoutPrediction, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import BaseLayoutOptions
from docling.models.base_layout_model import BaseLayoutModel
from docling_core.types.doc import DocItemLabel

from .options import Qwen3VlLayoutOptions, Qwen3VlQuantization
from .prompts import LAYOUT_ANALYSIS_PROMPT


_log = logging.getLogger(__name__)

_model_init_lock = threading.Lock()
_THINK_END_TOKEN_ID = 151668

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
        enabled: bool,
        artifacts_path: Path | None,
        options: Qwen3VlLayoutOptions,
        accelerator_options: AcceleratorOptions,
    ):
        self.enabled = enabled
        self.options = options
        self._processor = None
        self._model = None
        self._device = None

        if self.enabled:
            self._load_model()

    def _load_model(self) -> None:
        """Load the Qwen3-VL model."""
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Missing dependencies for Qwen3-VL. Install `torch` and `transformers>=4.51.0`."
            ) from exc

        requested_device = self.options.device or "cuda"
        if "cuda" not in requested_device:
            raise RuntimeError("Qwen3-VL requires a CUDA device.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device not available but required by Qwen3-VL.")

        torch_device = torch.device(requested_device)
        if torch_device.index is not None:
            torch.cuda.set_device(torch_device.index)

        torch_dtype = self._resolve_torch_dtype(self.options.dtype)

        with _model_init_lock:
            _log.info("Loading Qwen3-VL processor for layout analysis...")
            self._processor = AutoProcessor.from_pretrained(
                self.options.model_repo_id,
                trust_remote_code=self.options.trust_remote_code,
                token=self.options.hf_token,
            )

            attn_impl = self._select_attention_backend(self.options.attn_implementation)

            model_kwargs: dict = {
                "trust_remote_code": self.options.trust_remote_code,
                "device_map": "auto",
            }
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype
            if attn_impl:
                model_kwargs["attn_implementation"] = attn_impl
            if self.options.hf_token:
                model_kwargs["token"] = self.options.hf_token

            quantization_config = self._create_quantization_config()
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
                _log.info("Using %s quantization", self.options.quantization.value)

            _log.info("Loading Qwen3-VL model for layout analysis...")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.options.model_repo_id,
                **model_kwargs,
            )
            model = model.eval()

            self._model = model
            self._device = torch_device

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

        text_input = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            text=[text_input],
            images=[image_rgb],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.options.max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, input_len:]
        generated_ids = self._extract_after_think_token(generated_ids)

        output_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        self._maybe_empty_cache()

        # Parse JSON output
        try:
            json_match = re.search(r"\[[\s\S]*\]", output_text)
            if not json_match:
                _log.warning("No JSON array found in layout output")
                return []

            elements = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            _log.warning("Failed to parse layout JSON: %s", e)
            return []

        # Convert to Cluster objects
        return self._build_clusters(elements, page)

    def _build_clusters(self, elements: list[dict], page: Page) -> list[Cluster]:
        """Build Cluster objects from parsed JSON elements."""
        clusters = []

        for idx, elem in enumerate(elements):
            label_str = elem.get("label", "text").lower()
            bbox_data = elem.get("bbox", [0, 0, 1000, 1000])
            confidence = elem.get("confidence", 0.9)

            # Map label string to DocItemLabel
            label = LABEL_MAP.get(label_str, DocItemLabel.TEXT)

            # Convert bbox from 0-1000 scale to page coordinates
            if page.size and len(bbox_data) == 4:
                x1, y1, x2, y2 = bbox_data
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

    def _extract_after_think_token(self, generated_ids):
        """Extract tokens after </think> token for Thinking models."""
        ids = generated_ids[0].tolist()
        try:
            reversed_ids = ids[::-1]
            pos_from_end = reversed_ids.index(_THINK_END_TOKEN_ID)
            index = len(ids) - pos_from_end
        except ValueError:
            return generated_ids

        import torch

        return torch.tensor([ids[index:]], device=generated_ids.device)

    def _resolve_torch_dtype(self, dtype_name: str | None):
        if dtype_name is None or dtype_name == "auto":
            return "auto"
        try:
            import torch
        except ImportError:
            return None
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return mapping.get(dtype_name.lower(), getattr(torch, dtype_name.lower(), None))

    def _select_attention_backend(self, requested: str) -> str | None:
        if requested == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                _log.warning("flash-attn not installed; falling back to eager attention.")
                return "eager"
        return requested

    def _create_quantization_config(self):
        if self.options.quantization == Qwen3VlQuantization.NONE:
            return None
        try:
            import torch
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError("BitsAndBytes quantization requires `bitsandbytes` package.") from exc

        if self.options.quantization == Qwen3VlQuantization.INT8:
            return BitsAndBytesConfig(load_in_8bit=True)
        if self.options.quantization == Qwen3VlQuantization.INT4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.options.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.options.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        return None

    @staticmethod
    def _maybe_empty_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
