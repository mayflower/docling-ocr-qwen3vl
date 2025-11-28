"""Runtime wrapper around the Qwen3-VL model for OCR."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from PIL import Image

from .options import Qwen3VlOcrOptions, Qwen3VlPromptMode
from .prompts import resolve_prompt


_log = logging.getLogger(__name__)


@dataclass(slots=True)
class Qwen3VlResult:
    """Container for Qwen3-VL inference output."""

    text: str
    paragraphs: list[str]


class Qwen3VlRunner:
    """Lazy-load and execute the Qwen3-VL model for OCR."""

    def __init__(self, options: Qwen3VlOcrOptions):
        self.options = options
        self._processor = None
        self._model = None
        self._device = None

    def ensure_loaded(self) -> None:
        """Load the underlying model if not already available."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Missing dependencies for Qwen3-VL OCR integration. "
                "Install `torch` and `transformers>=4.51.0`."
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
        torch_dtype = _resolve_torch_dtype(self.options.dtype)

        processor_kwargs = {
            "trust_remote_code": self.options.trust_remote_code,
        }
        if self.options.hf_token:
            processor_kwargs["token"] = self.options.hf_token
        processor_kwargs.update(self.options.processor_kwargs)

        _log.info("Loading Qwen3-VL processor from %s", self.options.model_repo_id)
        self._processor = AutoProcessor.from_pretrained(
            self.options.model_repo_id, **processor_kwargs
        )

        attn_impl = _select_attention_backend(self.options.attn_implementation)

        model_kwargs = {
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
        quantization_config = _create_quantization_config(self.options)
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            _log.info(
                "Using %s quantization (this reduces VRAM usage)",
                self.options.quantization.value,
            )

        model_kwargs.update(self.options.model_kwargs)

        _log.info("Loading Qwen3-VL model (this can take a while)...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.options.model_repo_id,
            **model_kwargs,
        )
        model = model.eval()

        # If not using device_map, move to device manually (skip for quantized models)
        if model_kwargs.get("device_map") is None and quantization_config is None:
            model = model.to(torch_device)

        self._model = model
        self._device = torch_device
        self._attn_implementation = attn_impl

    def run(self, image: Image.Image, *, prompt_mode: Qwen3VlPromptMode) -> Qwen3VlResult:
        """Execute inference for a single image."""
        self.ensure_loaded()

        assert self._model is not None and self._processor is not None

        prompt = resolve_prompt(prompt_mode, self.options.prompt_overrides)
        image_rgb = image.convert("RGB")

        # Build chat messages for Qwen3-VL
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

        _log.debug(
            "Running Qwen3-VL OCR with prompt mode `%s` (max_new_tokens=%s)",
            prompt_mode.value,
            self.options.max_new_tokens,
        )

        # Generate with Qwen3-VL
        import torch

        with torch.no_grad():
            generated_ids = self._model.generate(
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
        output_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        output_text = output_text.strip()
        # Strip thinking tags from Qwen3-VL-Thinking models
        output_text = _strip_thinking_tags(output_text)
        paragraphs = _split_paragraphs(output_text)

        _maybe_empty_cache()

        return Qwen3VlResult(text=output_text, paragraphs=paragraphs)


def _strip_thinking_tags(text: str) -> str:
    """Remove thinking blocks from Qwen3-VL-Thinking model output."""
    import re

    # Remove everything between <think> and </think> tags (including tags)
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    # Also handle case where thinking starts implicitly and ends with </think>
    cleaned = re.sub(r"^.*?</think>\s*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs based on blank lines."""
    if not text:
        return []

    # Split on double newlines or more
    import re

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


def _resolve_torch_dtype(dtype_name: str | None):
    """Map option string to `torch.dtype` if possible."""
    if dtype_name is None or dtype_name == "auto":
        return "auto"

    try:
        import torch
    except ImportError:
        return None

    normalized = dtype_name.lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(normalized, getattr(torch, normalized, None))


def _select_attention_backend(requested: str) -> str | None:
    """Return an attention implementation that is supported in the environment."""
    if requested == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            _log.warning(
                "flash-attn is not installed; falling back to `attn_implementation='eager'`."
            )
            return "eager"
    return requested


def _maybe_empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _create_quantization_config(options: Qwen3VlOcrOptions):
    """Create BitsAndBytesConfig for quantization if requested."""
    from .options import Qwen3VlQuantization

    if options.quantization == Qwen3VlQuantization.NONE:
        return None

    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "BitsAndBytes quantization requires `bitsandbytes` package. "
            "Install with: pip install bitsandbytes"
        ) from exc

    if options.quantization == Qwen3VlQuantization.INT8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    if options.quantization == Qwen3VlQuantization.INT4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=options.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=options.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    return None
