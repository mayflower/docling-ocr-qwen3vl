"""Shared singleton registry for the Qwen3-VL model.

All plugin models (OCR, table structure, layout, picture description,
picture classifier, code/formula) share a single loaded model instance
keyed by (model_repo_id, device).  This avoids loading multiple copies
of the same ~8 GB model into GPU memory.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from .options import Qwen3VlQuantization

_log = logging.getLogger(__name__)

_lock = threading.Lock()
_instances: dict[tuple[str, str], SharedModel] = {}

# Token ID for </think> in Qwen3 Thinking models
THINK_END_TOKEN_ID = 151668


@dataclass(frozen=True, slots=True)
class SharedModel:
    """Handle returned by the registry — holds the loaded model, processor, and device."""

    model: object  # Qwen3VLForConditionalGeneration
    processor: object  # AutoProcessor
    device: object  # torch.device


def get_model(
    *,
    model_repo_id: str,
    device: str | None = "cuda",
    dtype: str | None = "bfloat16",
    trust_remote_code: bool = True,
    hf_token: str | None = None,
    attn_implementation: str = "flash_attention_2",
    quantization: Qwen3VlQuantization = Qwen3VlQuantization.NONE,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> SharedModel:
    """Return a shared model instance, loading it on first call.

    Thread-safe with double-checked locking.
    """
    requested_device = device or "cuda"
    key = (model_repo_id, requested_device)

    cached = _instances.get(key)
    if cached is not None:
        return cached

    with _lock:
        # Double-check after acquiring lock
        cached = _instances.get(key)
        if cached is not None:
            return cached

        shared = _load(
            model_repo_id=model_repo_id,
            requested_device=requested_device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            hf_token=hf_token,
            attn_implementation=attn_implementation,
            quantization=quantization,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        )
        _instances[key] = shared
        return shared


# -- internal helpers --------------------------------------------------------


def _load(
    *,
    model_repo_id: str,
    requested_device: str,
    dtype: str | None,
    trust_remote_code: bool,
    hf_token: str | None,
    attn_implementation: str,
    quantization: Qwen3VlQuantization,
    bnb_4bit_quant_type: str,
    bnb_4bit_use_double_quant: bool,
) -> SharedModel:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if "cuda" not in requested_device:
        raise RuntimeError("Qwen3-VL requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available but required by Qwen3-VL.")

    torch_device = torch.device(requested_device)
    if torch_device.index is not None:
        torch.cuda.set_device(torch_device.index)

    torch_dtype = resolve_torch_dtype(dtype)

    _log.info("Loading Qwen3-VL processor from %s", model_repo_id)
    processor = AutoProcessor.from_pretrained(
        model_repo_id,
        trust_remote_code=trust_remote_code,
        **({"token": hf_token} if hf_token else {}),
    )

    attn_impl = select_attention_backend(attn_implementation)

    model_kwargs: dict = {
        "trust_remote_code": trust_remote_code,
        "device_map": "auto" if torch_device.type == "cuda" else None,
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    if hf_token:
        model_kwargs["token"] = hf_token

    quant_config = create_quantization_config(
        quantization=quantization,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        _log.info("Using %s quantization", quantization.value)

    _log.info("Loading Qwen3-VL model (this can take a while)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_repo_id,
        **model_kwargs,
    )

    setattr(model, '_is_in_eval', True)  # noqa: B010
    model = model.eval()

    return SharedModel(model=model, processor=processor, device=torch_device)


def resolve_torch_dtype(dtype_name: str | None):  # type: ignore[no-untyped-def]
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
        "half": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(dtype_name.lower(), getattr(torch, dtype_name.lower(), None))


def select_attention_backend(requested: str) -> str | None:
    if requested == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            _log.warning(
                "flash-attn is not installed; falling back to eager attention."
            )
            return "eager"
    return requested


def create_quantization_config(
    *,
    quantization: Qwen3VlQuantization,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
):  # type: ignore[no-untyped-def]
    if quantization == Qwen3VlQuantization.NONE:
        return None
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "BitsAndBytes quantization requires `bitsandbytes` package."
        ) from exc

    if quantization == Qwen3VlQuantization.INT8:
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantization == Qwen3VlQuantization.INT4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return None


def extract_after_think_token(generated_ids):  # type: ignore[no-untyped-def]
    """Extract tokens after </think> token for Qwen3 Thinking models."""
    ids = generated_ids[0].tolist()
    try:
        reversed_ids = ids[::-1]
        pos_from_end = reversed_ids.index(THINK_END_TOKEN_ID)
        index = len(ids) - pos_from_end
    except ValueError:
        return generated_ids
    import torch

    return torch.tensor([ids[index:]], device=generated_ids.device)


def maybe_empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
