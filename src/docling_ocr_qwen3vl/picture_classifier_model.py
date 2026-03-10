"""Qwen3-VL Picture Classifier Model for Docling."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ItemAndImageEnrichmentElement
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling_core.types.doc import (
    DoclingDocument,
    NodeItem,
    PictureClassificationClass,
    PictureClassificationData,
    PictureItem,
)

from ._model_registry import extract_after_think_token, get_model, maybe_empty_cache
from .options import Qwen3VlPictureClassifierOptions
from .prompts import PICTURE_CLASSIFICATION_PROMPT


_log = logging.getLogger(__name__)


class Qwen3VlPictureClassifierModel(BaseItemAndImageEnrichmentModel):
    """Qwen3-VL based picture classification model.

    Uses the Qwen3-VL vision-language model to classify pictures/figures
    in documents (e.g., photograph, chart, diagram, illustration, etc.).
    """

    images_scale = 2.0

    def __init__(
        self,
        enabled: bool = True,
        artifacts_path: Path | None = None,
        options: Qwen3VlPictureClassifierOptions | None = None,
        accelerator_options: AcceleratorOptions | None = None,
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
        return self.enabled and isinstance(element, PictureItem)

    def __call__(
        self,
        doc: DoclingDocument,
        element_batch: Iterable[ItemAndImageEnrichmentElement],
    ) -> Iterable[NodeItem]:
        """Process a batch of pictures and add classification annotations."""
        if not self.enabled:
            for element in element_batch:
                yield element.item
            return

        for element in element_batch:
            if not isinstance(element.item, PictureItem):
                yield element.item
                continue

            # Classify the picture
            classification = self._classify_picture(element.image)

            if classification:
                element.item.annotations.append(classification)

            yield element.item

    def _classify_picture(self, image) -> PictureClassificationData | None:
        """Classify a picture using Qwen3-VL."""
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
                    {"type": "text", "text": PICTURE_CLASSIFICATION_PROMPT},
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
                _log.warning("No JSON found in classification output")
                return None

            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            _log.warning("Failed to parse classification JSON: %s", e)
            return None

        # Build classification result
        classes = data.get("classes", [])
        if not classes:
            return None

        predicted_classes = [
            PictureClassificationClass(
                class_name=c.get("class_name", "other"),
                confidence=c.get("confidence", 0.5),
            )
            for c in classes
        ]

        return PictureClassificationData(
            provenance="Qwen3VlPictureClassifier",
            predicted_classes=predicted_classes,
        )
