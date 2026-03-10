"""Qwen3-VL Picture Description Model for Docling enrichment pipeline."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import PictureDescriptionBaseOptions
from docling.models.picture_description_base_model import PictureDescriptionBaseModel
from PIL import Image

from ._model_registry import extract_after_think_token, get_model, maybe_empty_cache
from .options import Qwen3VlPictureDescriptionOptions


_log = logging.getLogger(__name__)


class Qwen3VlPictureDescriptionModel(PictureDescriptionBaseModel):
    """Qwen3-VL based picture description model for docling enrichment.

    This model uses the Qwen3-VL vision-language model to generate
    detailed descriptions of images/figures in documents. It supports
    quantization for reduced VRAM usage.
    """

    images_scale: float = 2.0

    def __init__(
        self,
        *,
        enabled: bool = True,
        enable_remote_services: bool = False,
        artifacts_path: Path | str | None = None,
        options: Qwen3VlPictureDescriptionOptions,
        accelerator_options: AcceleratorOptions,
        **kwargs,
    ):
        super().__init__(
            enabled=enabled,
            enable_remote_services=enable_remote_services,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: Qwen3VlPictureDescriptionOptions = options
        self.images_scale = options.scale
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
            self.provenance = f"qwen3vl:{self.options.model_repo_id}"

    @classmethod
    def get_options_type(cls) -> type[PictureDescriptionBaseOptions]:
        return Qwen3VlPictureDescriptionOptions

    def _annotate_images(self, images: Iterable[Image.Image]) -> Iterable[str]:
        """Generate descriptions for a batch of images."""
        import torch

        assert self._shared is not None
        model = self._shared.model
        processor = self._shared.processor

        for image in images:
            image_rgb = image.convert("RGB")

            # Build chat messages for Qwen3-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_rgb},
                        {"type": "text", "text": self.options.prompt},
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

            # Generate description
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.options.max_new_tokens,
                    temperature=self.options.temperature if self.options.do_sample else None,
                    do_sample=self.options.do_sample,
                )

            # Decode only the generated tokens (exclude input)
            input_len = inputs["input_ids"].shape[1]
            generated_ids = generated_ids[:, input_len:]

            # For Thinking models: extract content after </think> token
            generated_ids = extract_after_think_token(generated_ids)

            output_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0]

            # Clean up VRAM
            maybe_empty_cache()

            yield output_text.strip()
