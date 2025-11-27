# Docling Qwen3-VL OCR Plugin

This package integrates the [Qwen3-VL-8B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) vision-language model with [Docling](https://github.com/docling-project/docling) through the plugin system. It exposes Qwen3-VL as an OCR backend that can be selected in Docling pipelines or via the CLI.

## Features

- Powerful vision-language model with strong OCR capabilities
- Multilingual OCR support (32 languages)
- Native 256K context length
- Advanced document understanding and reasoning

## Requirements

- NVIDIA GPU with CUDA support (required)
- Python 3.10+
- Docling >= 2.57.0 with external plugins enabled
- transformers >= 4.51.0

## Installation

### From PyPI (not yet available)

```bash
pip install docling-ocr-qwen3vl
```

### From Source

```bash
git clone https://github.com/mayflower/docling-ocr-qwen3vl.git
cd docling-ocr-qwen3vl
pip install -e .
```

For development with tests:

```bash
pip install -e .[test]
```

## Usage

### Python

```python
from docling import DocumentConverter, InputFormat
from docling.datamodel.pipeline_options import PdfFormatOption, PdfPipelineOptions
from docling_ocr_qwen3vl.options import Qwen3VlOcrOptions

opts = PdfPipelineOptions()
opts.allow_external_plugins = True
opts.do_ocr = True
opts.ocr_options = Qwen3VlOcrOptions(force_full_page_ocr=True)

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
)
result = converter.convert("scanned.pdf")
print(result.document.export_to_markdown())
```

### CLI

```bash
docling --allow-external-plugins --ocr-engine qwen3vl_ocr scanned.pdf
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `model_repo_id` | `"Qwen/Qwen3-VL-8B-Thinking"` | Hugging Face model identifier |
| `device` | `"cuda"` | Torch device (cuda/cpu) |
| `dtype` | `"auto"` | Model dtype (auto, bfloat16, float16, float32) |
| `max_new_tokens` | `4096` | Maximum tokens to generate |
| `temperature` | `0.6` | Sampling temperature |
| `top_p` | `0.95` | Nucleus sampling probability |
| `top_k` | `20` | Top-k sampling parameter |
| `do_sample` | `True` | Enable stochastic decoding |
| `prompt_mode` | `OCR` | Prompt strategy (OCR, MARKDOWN, STRUCTURED) |
| `attn_implementation` | `"flash_attention_2"` | Attention backend |
| `page_scale` | `2.0` | PDF rasterization scale factor |

### Prompt Modes

- **OCR**: Extract plain text preserving reading order
- **MARKDOWN**: Convert document to markdown format
- **STRUCTURED**: Extract text with layout awareness (headings, paragraphs, tables)

## Attention Backends

Qwen3-VL defaults to `flash_attention_2`. If `flash-attn` is not available, the plugin automatically falls back to `eager` attention.

```python
Qwen3VlOcrOptions(attn_implementation="flash_attention_2")  # default, requires flash-attn
Qwen3VlOcrOptions(attn_implementation="eager")              # no flash-attn dependency
```

## Docling Integration Guide

1. **Install the plugin** alongside Docling (both must be in the same environment):

   ```bash
   pip install -U docling docling-ocr-qwen3vl
   ```

2. **Enable external plugins** in your pipeline options or CLI invocation:
   - Python: set `opts.allow_external_plugins = True`
   - CLI: pass `--allow-external-plugins`

3. **Select the OCR engine**:
   - Python: `opts.ocr_options = Qwen3VlOcrOptions(...)`
   - CLI: `docling --allow-external-plugins --ocr-engine qwen3vl_ocr scanned.pdf`

4. **Tune attention + device** (optional):
   - Use `Qwen3VlOcrOptions(attn_implementation="eager")` if flash-attn is absent
   - Adjust `Qwen3VlOcrOptions(device="cuda:1")` or `CUDA_VISIBLE_DEVICES` for multi-GPU

5. **Docling-serve**: set `DOCLING_SERVE_ALLOW_EXTERNAL_PLUGINS=1` and install this package in the same Python environment

## Development

- Run tests with `pytest`
- The implementation lives under `src/docling_ocr_qwen3vl/`

## License

MIT
