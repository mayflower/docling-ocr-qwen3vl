"""Pydantic options for the Qwen3-VL plugins."""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar, Literal

from docling.datamodel.pipeline_options import (
    BaseLayoutOptions,
    BaseTableStructureOptions,
    OcrOptions,
    PictureDescriptionBaseOptions,
)
from pydantic import BaseModel, ConfigDict, Field


class Qwen3VlPromptMode(str, Enum):
    """Available prompting strategies for Qwen3-VL OCR."""

    OCR = "ocr"
    MARKDOWN = "markdown"
    STRUCTURED = "structured"
    QWENVL_HTML = "qwenvl_html"


class Qwen3VlQuantization(str, Enum):
    """Quantization modes for Qwen3-VL."""

    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"


class Qwen3VlOcrOptions(OcrOptions):
    """Options exposed to Docling users for configuring the Qwen3-VL OCR engine."""

    kind: ClassVar[Literal["qwen3vl_ocr"]] = "qwen3vl_ocr"

    lang: list[str] = Field(
        default_factory=list,
        description=(
            "Qwen3-VL supports 32 languages without explicit configuration. "
            "Values provided here are ignored but accepted for compatibility."
        ),
    )
    model_repo_id: str = Field(
        default="Qwen/Qwen3-VL-8B-Thinking",
        description="Hugging Face repository identifier for the Qwen3-VL model.",
    )
    prompt_mode: Qwen3VlPromptMode = Field(
        default=Qwen3VlPromptMode.OCR,
        description="Prompt template selection that controls output format.",
    )
    prompt_overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Override specific prompt templates; keys correspond to prompt modes.",
    )
    device: str | None = Field(
        default="cuda",
        description="Torch device string; defaults to GPU when available.",
    )
    dtype: str = Field(
        default="bfloat16",
        description="Torch dtype passed to the model initialization.",
    )
    max_new_tokens: int = Field(
        default=4096,
        description="Maximum tokens generated during inference.",
        ge=1,
    )
    temperature: float = Field(
        default=0.6,
        description="Sampling temperature for decoding.",
        ge=0.0,
    )
    top_p: float = Field(
        default=0.95,
        description="Top-p (nucleus) sampling probability.",
        ge=0.0,
        le=1.0,
    )
    top_k: int = Field(
        default=20,
        description="Top-k sampling parameter.",
        ge=1,
    )
    do_sample: bool = Field(
        default=True,
        description="Enable stochastic decoding.",
    )
    timeout_s: float | None = Field(
        default=None,
        description="Optional timeout for inference calls.",
        ge=0.0,
    )
    hf_token: str | None = Field(
        default=None,
        description="Optional Hugging Face token for gated models.",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Allow loading custom model code from the repository.",
    )
    processor_kwargs: dict[str, object] = Field(
        default_factory=dict,
        description="Additional kwargs forwarded to `AutoProcessor.from_pretrained`.",
    )
    model_kwargs: dict[str, object] = Field(
        default_factory=dict,
        description="Additional kwargs forwarded to the underlying model loader.",
    )
    attn_implementation: str = Field(
        default="flash_attention_2",
        description="Attention implementation hint passed to the model.",
    )
    page_scale: float = Field(
        default=2.0,
        description="Scale factor applied when rasterizing PDF regions for inference.",
        ge=1.0,
        le=4.0,
    )
    quantization: Qwen3VlQuantization = Field(
        default=Qwen3VlQuantization.NONE,
        description="Quantization mode: none (full precision), int8 (8-bit), int4 (4-bit).",
    )
    bnb_4bit_quant_type: str = Field(
        default="nf4",
        description="4-bit quantization type for BitsAndBytes: 'nf4' or 'fp4'.",
    )
    bnb_4bit_use_double_quant: bool = Field(
        default=True,
        description="Use nested quantization for additional memory savings.",
    )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )


class Qwen3VlPictureDescriptionOptions(PictureDescriptionBaseOptions):
    """Options for Qwen3-VL picture description (image captioning/enrichment)."""

    kind: ClassVar[Literal["qwen3vl"]] = "qwen3vl"

    model_repo_id: str = Field(
        default="Qwen/Qwen3-VL-8B-Thinking",
        description="Hugging Face repository identifier for the Qwen3-VL model.",
    )
    prompt: str = Field(
        default="Describe this image in detail. Include information about the type of content (photo, chart, diagram, illustration, etc.), the main subject, and any relevant details visible in the image.",
        description="Prompt template for image description.",
    )
    device: str | None = Field(
        default="cuda",
        description="Torch device string; defaults to GPU when available.",
    )
    dtype: str = Field(
        default="bfloat16",
        description="Torch dtype passed to the model initialization.",
    )
    max_new_tokens: int = Field(
        default=512,
        description="Maximum tokens generated during inference.",
        ge=1,
    )
    temperature: float = Field(
        default=0.6,
        description="Sampling temperature for decoding.",
        ge=0.0,
    )
    do_sample: bool = Field(
        default=True,
        description="Enable stochastic decoding.",
    )
    hf_token: str | None = Field(
        default=None,
        description="Optional Hugging Face token for gated models.",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Allow loading custom model code from the repository.",
    )
    attn_implementation: str = Field(
        default="flash_attention_2",
        description="Attention implementation hint passed to the model.",
    )
    quantization: Qwen3VlQuantization = Field(
        default=Qwen3VlQuantization.NONE,
        description="Quantization mode: none (full precision), int8 (8-bit), int4 (4-bit).",
    )
    bnb_4bit_quant_type: str = Field(
        default="nf4",
        description="4-bit quantization type for BitsAndBytes: 'nf4' or 'fp4'.",
    )
    bnb_4bit_use_double_quant: bool = Field(
        default=True,
        description="Use nested quantization for additional memory savings.",
    )
    generation_config: dict[str, Any] = Field(
        default_factory=lambda: {"max_new_tokens": 512, "do_sample": True, "temperature": 0.6},
        description="Generation config passed to model.generate().",
    )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )


class Qwen3VlTableStructureOptions(BaseTableStructureOptions):
    """Options for Qwen3-VL table structure detection."""

    kind: ClassVar[Literal["qwen3vl_table"]] = "qwen3vl_table"

    model_repo_id: str = Field(
        default="Qwen/Qwen3-VL-8B-Thinking",
        description="Hugging Face repository identifier for the Qwen3-VL model.",
    )
    device: str | None = Field(
        default="cuda",
        description="Torch device string; defaults to GPU when available.",
    )
    dtype: str = Field(
        default="bfloat16",
        description="Torch dtype passed to the model initialization.",
    )
    max_new_tokens: int = Field(
        default=4096,
        description="Maximum tokens generated during inference.",
        ge=1,
    )
    hf_token: str | None = Field(
        default=None,
        description="Optional Hugging Face token for gated models.",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Allow loading custom model code from the repository.",
    )
    attn_implementation: str = Field(
        default="flash_attention_2",
        description="Attention implementation hint passed to the model.",
    )
    quantization: Qwen3VlQuantization = Field(
        default=Qwen3VlQuantization.NONE,
        description="Quantization mode: none (full precision), int8 (8-bit), int4 (4-bit).",
    )
    bnb_4bit_quant_type: str = Field(
        default="nf4",
        description="4-bit quantization type for BitsAndBytes: 'nf4' or 'fp4'.",
    )
    bnb_4bit_use_double_quant: bool = Field(
        default=True,
        description="Use nested quantization for additional memory savings.",
    )
    do_cell_matching: bool = Field(
        default=True,
        description="Match predicted cells back to PDF text cells.",
    )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )


class Qwen3VlLayoutOptions(BaseLayoutOptions):
    """Options for Qwen3-VL layout analysis."""

    kind: ClassVar[Literal["qwen3vl_layout"]] = "qwen3vl_layout"

    model_repo_id: str = Field(
        default="Qwen/Qwen3-VL-8B-Thinking",
        description="Hugging Face repository identifier for the Qwen3-VL model.",
    )
    device: str | None = Field(
        default="cuda",
        description="Torch device string; defaults to GPU when available.",
    )
    dtype: str = Field(
        default="bfloat16",
        description="Torch dtype passed to the model initialization.",
    )
    max_new_tokens: int = Field(
        default=4096,
        description="Maximum tokens generated during inference.",
        ge=1,
    )
    hf_token: str | None = Field(
        default=None,
        description="Optional Hugging Face token for gated models.",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Allow loading custom model code from the repository.",
    )
    attn_implementation: str = Field(
        default="flash_attention_2",
        description="Attention implementation hint passed to the model.",
    )
    quantization: Qwen3VlQuantization = Field(
        default=Qwen3VlQuantization.NONE,
        description="Quantization mode: none (full precision), int8 (8-bit), int4 (4-bit).",
    )
    bnb_4bit_quant_type: str = Field(
        default="nf4",
        description="4-bit quantization type for BitsAndBytes: 'nf4' or 'fp4'.",
    )
    bnb_4bit_use_double_quant: bool = Field(
        default=True,
        description="Use nested quantization for additional memory savings.",
    )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )


class Qwen3VlPictureClassifierOptions(BaseModel):
    """Options for Qwen3-VL picture classification."""

    kind: ClassVar[Literal["qwen3vl_classifier"]] = "qwen3vl_classifier"

    model_repo_id: str = Field(
        default="Qwen/Qwen3-VL-8B-Thinking",
        description="Hugging Face repository identifier for the Qwen3-VL model.",
    )
    device: str | None = Field(
        default="cuda",
        description="Torch device string; defaults to GPU when available.",
    )
    dtype: str = Field(
        default="bfloat16",
        description="Torch dtype passed to the model initialization.",
    )
    max_new_tokens: int = Field(
        default=256,
        description="Maximum tokens generated during inference.",
        ge=1,
    )
    hf_token: str | None = Field(
        default=None,
        description="Optional Hugging Face token for gated models.",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Allow loading custom model code from the repository.",
    )
    attn_implementation: str = Field(
        default="flash_attention_2",
        description="Attention implementation hint passed to the model.",
    )
    quantization: Qwen3VlQuantization = Field(
        default=Qwen3VlQuantization.NONE,
        description="Quantization mode: none (full precision), int8 (8-bit), int4 (4-bit).",
    )
    bnb_4bit_quant_type: str = Field(
        default="nf4",
        description="4-bit quantization type for BitsAndBytes: 'nf4' or 'fp4'.",
    )
    bnb_4bit_use_double_quant: bool = Field(
        default=True,
        description="Use nested quantization for additional memory savings.",
    )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )


class Qwen3VlCodeFormulaOptions(BaseModel):
    """Options for Qwen3-VL code and formula detection."""

    kind: ClassVar[Literal["qwen3vl_code_formula"]] = "qwen3vl_code_formula"

    model_repo_id: str = Field(
        default="Qwen/Qwen3-VL-8B-Thinking",
        description="Hugging Face repository identifier for the Qwen3-VL model.",
    )
    device: str | None = Field(
        default="cuda",
        description="Torch device string; defaults to GPU when available.",
    )
    dtype: str = Field(
        default="bfloat16",
        description="Torch dtype passed to the model initialization.",
    )
    max_new_tokens: int = Field(
        default=2048,
        description="Maximum tokens generated during inference.",
        ge=1,
    )
    hf_token: str | None = Field(
        default=None,
        description="Optional Hugging Face token for gated models.",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Allow loading custom model code from the repository.",
    )
    attn_implementation: str = Field(
        default="flash_attention_2",
        description="Attention implementation hint passed to the model.",
    )
    quantization: Qwen3VlQuantization = Field(
        default=Qwen3VlQuantization.NONE,
        description="Quantization mode: none (full precision), int8 (8-bit), int4 (4-bit).",
    )
    bnb_4bit_quant_type: str = Field(
        default="nf4",
        description="4-bit quantization type for BitsAndBytes: 'nf4' or 'fp4'.",
    )
    bnb_4bit_use_double_quant: bool = Field(
        default=True,
        description="Use nested quantization for additional memory savings.",
    )
    do_code_enrichment: bool = Field(
        default=True,
        description="Enable code block detection and extraction.",
    )
    do_formula_enrichment: bool = Field(
        default=True,
        description="Enable formula detection and LaTeX extraction.",
    )

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )
