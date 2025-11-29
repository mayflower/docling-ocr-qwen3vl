"""Qwen3-VL Picture Description Model for Docling enrichment pipeline."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import PictureDescriptionBaseOptions
from docling.models.picture_description_base_model import PictureDescriptionBaseModel
from PIL import Image

from .options import Qwen3VlPictureDescriptionOptions, Qwen3VlQuantization


_log = logging.getLogger(__name__)

# Global lock for model initialization to prevent threading issues
_model_init_lock = threading.Lock()

# Token ID for </think> in Qwen3 Thinking models
_THINK_END_TOKEN_ID = 151668


class Qwen3VlPictureDescriptionModel(PictureDescriptionBaseModel):
    """Qwen3-VL based picture description model for docling enrichment.

    This model uses the Qwen3-VL vision-language model to generate
    detailed descriptions of images/figures in documents. It supports
    quantization for reduced VRAM usage.
    """

    images_scale: float = 2.0

    def __init__(
        self,
        enabled: bool,
        enable_remote_services: bool,
        artifacts_path: Path | str | None,
        options: Qwen3VlPictureDescriptionOptions,
        accelerator_options: AcceleratorOptions,
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

        self._processor = None
        self._model = None
        self._device = None

        if self.enabled:
            self._load_model(accelerator_options)
            self.provenance = f"qwen3vl:{self.options.model_repo_id}"

    def _load_model(self, accelerator_options: AcceleratorOptions) -> None:
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
            raise RuntimeError(
                "Qwen3-VL requires a CUDA device. "
                "Set `device='cuda'` and ensure an NVIDIA GPU is available."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device not available but required by Qwen3-VL. "
                "Check GPU visibility and drivers."
            )

        torch_device = torch.device(requested_device)
        if torch_device.index is not None:
            torch.cuda.set_device(torch_device.index)

        # Resolve dtype
        torch_dtype = self._resolve_torch_dtype(self.options.dtype)

        with _model_init_lock:
            _log.info("Loading Qwen3-VL processor from %s", self.options.model_repo_id)
            self._processor = AutoProcessor.from_pretrained(
                self.options.model_repo_id,
                trust_remote_code=self.options.trust_remote_code,
                token=self.options.hf_token,
            )

            attn_impl = self._select_attention_backend(self.options.attn_implementation)

            model_kwargs: dict = {
                "trust_remote_code": self.options.trust_remote_code,
                "device_map": "auto" if torch_device.type == "cuda" else None,
            }
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype
            if attn_impl:
                model_kwargs["attn_implementation"] = attn_impl
            if self.options.hf_token:
                model_kwargs["token"] = self.options.hf_token

            # Configure quantization if requested
            quantization_config = self._create_quantization_config()
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
                _log.info(
                    "Using %s quantization for picture description",
                    self.options.quantization.value,
                )

            _log.info("Loading Qwen3-VL model for picture description...")
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.options.model_repo_id,
                **model_kwargs,
            )
            model = model.eval()

            self._model = model
            self._device = torch_device

    @classmethod
    def get_options_type(cls) -> type[PictureDescriptionBaseOptions]:
        return Qwen3VlPictureDescriptionOptions

    def _annotate_images(self, images: Iterable[Image.Image]) -> Iterable[str]:
        """Generate descriptions for a batch of images."""
        import torch

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

            # Move inputs to model device
            inputs = inputs.to(self._model.device)

            # Generate description
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.options.max_new_tokens,
                    temperature=self.options.temperature if self.options.do_sample else None,
                    do_sample=self.options.do_sample,
                )

            # Decode only the generated tokens (exclude input)
            input_len = inputs["input_ids"].shape[1]
            generated_ids = generated_ids[:, input_len:]

            # For Thinking models: extract content after </think> token
            generated_ids = self._extract_after_think_token(generated_ids)

            output_text = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0]

            # Clean up VRAM
            self._maybe_empty_cache()

            yield output_text.strip()

    def _extract_after_think_token(self, generated_ids):  # type: ignore[no-untyped-def]
        """Extract tokens after </think> token for Qwen3-VL-Thinking models."""
        ids = generated_ids[0].tolist()

        try:
            reversed_ids = ids[::-1]
            pos_from_end = reversed_ids.index(_THINK_END_TOKEN_ID)
            index = len(ids) - pos_from_end
        except ValueError:
            # No </think> token found - not a Thinking model output
            return generated_ids

        import torch

        return torch.tensor([ids[index:]], device=generated_ids.device)

    def _resolve_torch_dtype(self, dtype_name: str | None):  # type: ignore[no-untyped-def]
        """Map option string to torch.dtype."""
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
        """Return an attention implementation that is supported."""
        if requested == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                _log.warning("flash-attn not installed; falling back to eager attention.")
                return "eager"
        return requested

    def _create_quantization_config(self):  # type: ignore[no-untyped-def]
        """Create BitsAndBytesConfig for quantization if requested."""
        if self.options.quantization == Qwen3VlQuantization.NONE:
            return None

        try:
            import torch
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "BitsAndBytes quantization requires `bitsandbytes` package. "
                "Install with: pip install bitsandbytes"
            ) from exc

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
        """Clear CUDA cache to free memory."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
