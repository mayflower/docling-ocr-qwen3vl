"""Qwen3-VL Code and Formula Model for Docling."""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Iterable
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling_core.types.doc import CodeItem, DocItemLabel, DoclingDocument, NodeItem, TextItem
from docling_core.types.doc.labels import CodeLanguageLabel

from .options import Qwen3VlCodeFormulaOptions, Qwen3VlQuantization
from .prompts import CODE_DETECTION_PROMPT


_log = logging.getLogger(__name__)

_model_init_lock = threading.Lock()
_THINK_END_TOKEN_ID = 151668

# Map language strings to CodeLanguageLabel
LANGUAGE_MAP: dict[str, CodeLanguageLabel] = {
    "python": CodeLanguageLabel.PYTHON,
    "javascript": CodeLanguageLabel.JAVASCRIPT,
    "typescript": CodeLanguageLabel.TYPESCRIPT,
    "java": CodeLanguageLabel.JAVA,
    "c": CodeLanguageLabel.C,
    "cpp": CodeLanguageLabel.C_PLUS_PLUS,
    "c++": CodeLanguageLabel.C_PLUS_PLUS,
    "csharp": CodeLanguageLabel.C_SHARP,
    "c#": CodeLanguageLabel.C_SHARP,
    "go": CodeLanguageLabel.GO,
    "rust": CodeLanguageLabel.RUST,
    "ruby": CodeLanguageLabel.RUBY,
    "php": CodeLanguageLabel.PHP,
    "swift": CodeLanguageLabel.SWIFT,
    "kotlin": CodeLanguageLabel.KOTLIN,
    "sql": CodeLanguageLabel.SQL,
    "bash": CodeLanguageLabel.BASH,
    "shell": CodeLanguageLabel.BASH,
    "html": CodeLanguageLabel.HTML,
    "css": CodeLanguageLabel.CSS,
    "json": CodeLanguageLabel.JSON,
    "yaml": CodeLanguageLabel.YAML,
    "xml": CodeLanguageLabel.XML,
}


class Qwen3VlCodeFormulaModel(BaseItemAndImageEnrichmentModel):
    """Qwen3-VL based code and formula detection model.

    Uses the Qwen3-VL vision-language model to detect and extract
    code blocks and mathematical formulas from document images.
    """

    images_scale = 1.67  # 120 DPI aligned with training data
    elements_batch_size = 5

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
        options: Qwen3VlCodeFormulaOptions,
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
            _log.info("Loading Qwen3-VL processor for code/formula detection...")
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

            _log.info("Loading Qwen3-VL model for code/formula detection...")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.options.model_repo_id,
                **model_kwargs,
            )
            model = model.eval()

            self._model = model
            self._device = torch_device

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        """Check if element can be processed by this model."""
        if not self.enabled:
            return False

        # Process CodeItem elements if code enrichment is enabled
        if isinstance(element, CodeItem) and self.options.do_code_enrichment:
            return True

        # Process TextItem elements with FORMULA label if formula enrichment is enabled
        if (
            isinstance(element, TextItem)
            and element.label == DocItemLabel.FORMULA
            and self.options.do_formula_enrichment
        ):
            return True

        return False

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        """Process a batch of elements and enrich with code/formula content."""
        if not self.enabled:
            for element in element_batch:
                yield element.item
            return

        for element in element_batch:
            item = element.item
            image = element.image

            if isinstance(item, CodeItem) and self.options.do_code_enrichment:
                # Extract code content
                result = self._extract_code_or_formula(image)
                if result and result.get("is_code"):
                    item.text = result.get("code", item.text or "")
                    lang_str = result.get("language", "").lower()
                    item.code_language = LANGUAGE_MAP.get(lang_str, CodeLanguageLabel.UNKNOWN)

            elif (
                isinstance(item, TextItem)
                and item.label == DocItemLabel.FORMULA
                and self.options.do_formula_enrichment
            ):
                # Extract formula content
                result = self._extract_code_or_formula(image)
                if result and result.get("is_formula"):
                    item.text = result.get("latex", item.text or "")

            yield item

    def _extract_code_or_formula(self, image) -> dict | None:
        """Extract code or formula from an image using Qwen3-VL."""
        import torch

        image_rgb = image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_rgb},
                    {"type": "text", "text": CODE_DETECTION_PROMPT},
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
            json_match = re.search(r"\{[\s\S]*\}", output_text)
            if not json_match:
                _log.warning("No JSON found in code/formula output")
                return None

            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            _log.warning("Failed to parse code/formula JSON: %s", e)
            return None

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
