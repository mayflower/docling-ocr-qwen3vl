"""Qwen3-VL Code and Formula Model for Docling."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling_core.types.doc import CodeItem, DocItemLabel, DoclingDocument, NodeItem, TextItem
from docling_core.types.doc.labels import CodeLanguageLabel

from ._model_registry import extract_after_think_token, get_model, maybe_empty_cache
from .options import Qwen3VlCodeFormulaOptions
from .prompts import CODE_DETECTION_PROMPT


_log = logging.getLogger(__name__)

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

        assert self._shared is not None
        model = self._shared.model
        processor = self._shared.processor

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
