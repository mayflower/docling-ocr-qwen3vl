"""Preset registration that bridges Qwen3-VL into docling's new VLM engine framework.

Without this, `picture_description_preset: "qwen3vl"` is rejected by docling-serve
because the new picture-description architecture (PictureDescriptionVlmEngineOptions
with model_spec + engine_options) doesn't dispatch to the plugin's factory by kind.
"""

from __future__ import annotations

from docling.datamodel.pipeline_options import PictureDescriptionVlmEngineOptions
from docling.datamodel.pipeline_options_vlm_model import (
    ResponseFormat,
    TransformersModelType,
)
from docling.datamodel.stage_model_specs import (
    EngineModelConfig,
    StageModelPreset,
    VlmModelSpec,
)
from docling.models.inference_engines.vlm.base import VlmEngineType

from .options import DEFAULT_QWEN3VL_MODEL_REPO_ID


# --- transformers 4.57.x repr bug workaround --------------------------------
# transformers.configuration_utils.PretrainedConfig.to_dict() unconditionally
# calls self.quantization_config.to_dict() (line 964 in 4.57.6). For some
# AutoConfig load paths the attribute exists but is None, so a simple
# `logger.info(f"Model config {config}")` in from_dict crashes with
# AttributeError. Hits us on unsloth/Qwen3-VL-4B-Instruct-bnb-4bit. Trivial
# patch: guard the call. Safe to apply unconditionally.
def _install_pretrained_config_to_dict_patch() -> None:
    from transformers import configuration_utils as _cu

    if getattr(_cu.PretrainedConfig.to_dict, "_qwen3vl_patched", False):
        return

    def _wrap(method_name: str):  # type: ignore[no-untyped-def]
        orig = getattr(_cu.PretrainedConfig, method_name)

        def _safe(self):  # type: ignore[no-untyped-def]
            swapped = (
                hasattr(self, "quantization_config")
                and self.quantization_config is None
            )
            if swapped:
                try:
                    object.__setattr__(self, "quantization_config", {})
                except Exception:
                    swapped = False
            try:
                return orig(self)
            finally:
                if swapped:
                    try:
                        object.__setattr__(self, "quantization_config", None)
                    except Exception:
                        pass

        _safe._qwen3vl_patched = True  # type: ignore[attr-defined]
        setattr(_cu.PretrainedConfig, method_name, _safe)

    # Both to_dict and to_diff_dict have the same blind
    # self.quantization_config.to_dict() bug in transformers 4.57.x.
    _wrap("to_dict")
    _wrap("to_diff_dict")


_install_pretrained_config_to_dict_patch()


QWEN3VL_PICTURE_DESC = StageModelPreset(
    preset_id="qwen3vl",
    name="Qwen3-VL-4B-Instruct (bnb-4bit)",
    description=(
        "Qwen3-VL 4B Instruct (unsloth bnb-4bit) for image descriptions. "
        "Picks the same model the plugin already loads for OCR so the two "
        "stages share the same architecture; the model still loads twice "
        "into VRAM because the new engine framework has its own loader."
    ),
    model_spec=VlmModelSpec(
        name="Qwen3-VL-4B-Instruct-bnb-4bit",
        default_repo_id=DEFAULT_QWEN3VL_MODEL_REPO_ID,
        prompt="Describe this image in detail.",
        response_format=ResponseFormat.PLAINTEXT,
        supported_engines={VlmEngineType.TRANSFORMERS},
        engine_overrides={
            VlmEngineType.TRANSFORMERS: EngineModelConfig(
                torch_dtype="bfloat16",
                extra_config={
                    "transformers_model_type": (
                        TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT
                    ),
                    # Model ships its own bnb-4bit config in config.json; let
                    # transformers pick it up. Don't set quantized=True or
                    # docling stacks BitsAndBytesConfig(8bit) on top and breaks.
                    "quantized": False,
                    "trust_remote_code": True,
                },
            ),
        },
        trust_remote_code=True,
        max_new_tokens=512,
    ),
    scale=2.0,
    default_engine_type=VlmEngineType.TRANSFORMERS,
    stage_options={
        "picture_area_threshold": 0.05,
    },
)


def register_presets() -> None:
    """Register Qwen3-VL presets with docling's stage preset registries."""
    PictureDescriptionVlmEngineOptions.register_preset(QWEN3VL_PICTURE_DESC)


register_presets()

# The kind-based plugin escape hatch in docling-jobkit's
# DoclingConverterManager._parse_picture_description_options
# (mayflower fork) routes
# picture_description_custom_config={"kind": "qwen3vl"} requests to the
# plugin's Qwen3VlPictureDescriptionOptions / Qwen3VlPictureDescriptionModel,
# reusing the shared singleton already loaded by OCR/layout/tables.
