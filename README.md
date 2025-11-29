# Docling Qwen3-VL OCR Plugin

This package integrates the [Qwen3-VL-8B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) vision-language model with [Docling](https://github.com/docling-project/docling) through the plugin system. It exposes Qwen3-VL as an OCR backend that can be selected in Docling pipelines or via the CLI.

## Features

- Powerful vision-language model with strong OCR capabilities
- **Layout-aware OCR with bounding boxes** via QWENVL_HTML mode
- Multilingual OCR support (32 languages)
- Native 256K context length
- Advanced document understanding and reasoning
- Support for the "Thinking" model variant with chain-of-thought reasoning

## Requirements

- **NVIDIA GPU with CUDA support** (required - CPU not supported)
- Python 3.10+
- Docling >= 2.57.0 with external plugins enabled
- transformers >= 4.51.0
- ~16GB VRAM (full precision) or ~5GB VRAM (4-bit quantization)

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

For quantization support (reduces VRAM from ~16GB to ~5GB):

```bash
pip install -e ".[quantization]"
```

For development with tests:

```bash
pip install -e ".[test,quantization]"
```

## Quick Start

### Python API

```python
from docling import DocumentConverter, InputFormat
from docling.datamodel.pipeline_options import PdfFormatOption, PdfPipelineOptions
from docling_ocr_qwen3vl.options import Qwen3VlOcrOptions, Qwen3VlPromptMode, Qwen3VlQuantization

# Configure pipeline with Qwen3-VL OCR
opts = PdfPipelineOptions()
opts.allow_external_plugins = True
opts.do_ocr = True
opts.ocr_options = Qwen3VlOcrOptions(
    prompt_mode=Qwen3VlPromptMode.QWENVL_HTML,  # Layout-aware with bounding boxes
    quantization=Qwen3VlQuantization.INT4,       # Use 4-bit to reduce VRAM
    force_full_page_ocr=True,
)

# Convert document
converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
)
result = converter.convert("scanned.pdf")
print(result.document.export_to_markdown())
```

### CLI

```bash
# Basic usage
docling --allow-external-plugins --ocr-engine qwen3vl_ocr scanned.pdf

# With specific options (JSON format)
docling --allow-external-plugins \
  --ocr-engine qwen3vl_ocr \
  --ocr-options '{"prompt_mode": "qwenvl_html", "quantization": "int4"}' \
  scanned.pdf
```

## Prompt Modes

The plugin supports multiple prompt modes for different use cases:

| Mode | Description | Bounding Boxes |
|------|-------------|----------------|
| `OCR` | Extract plain text preserving reading order | No |
| `MARKDOWN` | Convert document to markdown format | No |
| `STRUCTURED` | Extract text with layout awareness | No |
| **`QWENVL_HTML`** | **Layout-aware HTML with precise bounding boxes** | **Yes** |

### QWENVL_HTML Mode (Recommended for Docling)

The `QWENVL_HTML` mode produces HTML output with `data-bbox` attributes containing element coordinates. This is the recommended mode for Docling integration as it provides:

- Accurate bounding boxes for each text element
- Proper element type classification (headings, paragraphs, tables, etc.)
- Normalized coordinates (0-1000 scale) that Docling converts to document coordinates

```python
from docling_ocr_qwen3vl.options import Qwen3VlOcrOptions, Qwen3VlPromptMode

opts.ocr_options = Qwen3VlOcrOptions(
    prompt_mode=Qwen3VlPromptMode.QWENVL_HTML,
    force_full_page_ocr=True,
)
```

Example output format:
```html
<h1 data-bbox="400 80 580 90">Document Title</h1>
<p data-bbox="100 120 900 150">First paragraph text...</p>
<p data-bbox="100 160 900 190">Second paragraph text...</p>
```

## Quantization (Reduce VRAM Usage)

The plugin supports 4-bit and 8-bit quantization via BitsAndBytes, significantly reducing VRAM requirements:

| Mode | VRAM (approx) | Quality |
|------|---------------|---------|
| Full precision (bf16) | ~16GB | Best |
| 8-bit (int8) | ~8GB | Very Good |
| **4-bit (int4)** | **~5GB** | Good |

### Installation

```bash
pip install -e ".[quantization]"
# or
pip install bitsandbytes
```

### Usage

```python
from docling_ocr_qwen3vl.options import Qwen3VlOcrOptions, Qwen3VlQuantization

# 4-bit quantization (recommended for limited VRAM)
opts.ocr_options = Qwen3VlOcrOptions(
    quantization=Qwen3VlQuantization.INT4,
    force_full_page_ocr=True,
)

# 8-bit quantization (better quality, more VRAM)
opts.ocr_options = Qwen3VlOcrOptions(
    quantization=Qwen3VlQuantization.INT8,
    force_full_page_ocr=True,
)
```

### Quantization Options

| Option | Default | Description |
|--------|---------|-------------|
| `quantization` | `NONE` | Quantization mode: `NONE`, `INT8`, `INT4` |
| `bnb_4bit_quant_type` | `"nf4"` | 4-bit quantization type: `nf4` or `fp4` |
| `bnb_4bit_use_double_quant` | `True` | Nested quantization for extra memory savings |

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `model_repo_id` | `"Qwen/Qwen3-VL-8B-Thinking"` | Hugging Face model identifier |
| `device` | `"cuda"` | Torch device (must be CUDA) |
| `dtype` | `"auto"` | Model dtype (auto, bfloat16, float16, float32) |
| `max_new_tokens` | `4096` | Maximum tokens to generate |
| `temperature` | `0.6` | Sampling temperature |
| `top_p` | `0.95` | Nucleus sampling probability |
| `top_k` | `20` | Top-k sampling parameter |
| `do_sample` | `True` | Enable stochastic decoding |
| `prompt_mode` | `OCR` | Prompt strategy (OCR, MARKDOWN, STRUCTURED, QWENVL_HTML) |
| `attn_implementation` | `"flash_attention_2"` | Attention backend |
| `page_scale` | `2.0` | PDF rasterization scale factor |

## Docling Integration

### Step 1: Install Both Packages

Both Docling and this plugin must be in the same Python environment:

```bash
pip install -U docling
pip install -e ".[quantization]"  # or pip install docling-ocr-qwen3vl
```

### Step 2: Enable External Plugins

External plugins must be explicitly enabled for security:

**Python:**
```python
opts = PdfPipelineOptions()
opts.allow_external_plugins = True
```

**CLI:**
```bash
docling --allow-external-plugins ...
```

### Step 3: Configure OCR Options

**Python (recommended configuration):**
```python
from docling_ocr_qwen3vl.options import Qwen3VlOcrOptions, Qwen3VlPromptMode, Qwen3VlQuantization

opts.do_ocr = True
opts.ocr_options = Qwen3VlOcrOptions(
    prompt_mode=Qwen3VlPromptMode.QWENVL_HTML,  # Best for layout analysis
    quantization=Qwen3VlQuantization.INT4,       # Reduce VRAM usage
    force_full_page_ocr=True,                    # OCR entire page
)
```

**CLI:**
```bash
docling --allow-external-plugins \
  --ocr-engine qwen3vl_ocr \
  --ocr-options '{"prompt_mode": "qwenvl_html", "quantization": "int4", "force_full_page_ocr": true}' \
  scanned.pdf
```

## Docling-Serve Integration

[Docling-serve](https://github.com/docling-project/docling-serve) is the HTTP API server for Docling. To use this plugin with docling-serve:

### Step 1: Install in the Same Environment

```bash
# Install docling-serve
pip install docling-serve

# Install this plugin
pip install docling-ocr-qwen3vl[quantization]
# or from source:
pip install -e ".[quantization]"
```

### Step 2: Enable External Plugins

Set the environment variable before starting the server:

```bash
export DOCLING_SERVE_ALLOW_EXTERNAL_PLUGINS=1
```

### Step 3: Start the Server

```bash
# Start docling-serve
docling-serve run

# Or with uvicorn directly
uvicorn docling_serve.app:app --host 0.0.0.0 --port 5000
```

### Step 4: Make API Requests

Use the `/convert` endpoint with OCR options in the request body:

```bash
curl -X POST "http://localhost:5000/convert" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@scanned.pdf" \
  -F 'options={
    "pdf_pipeline_options": {
      "do_ocr": true,
      "ocr_options": {
        "kind": "qwen3vl_ocr",
        "prompt_mode": "qwenvl_html",
        "quantization": "int4",
        "force_full_page_ocr": true
      }
    }
  }'
```

### Docker Deployment

For Docker deployments, ensure the plugin is installed in the container and the environment variable is set:

```dockerfile
FROM python:3.11-slim

# Install CUDA runtime (required)
# ... CUDA installation steps ...

# Install dependencies
RUN pip install docling-serve docling-ocr-qwen3vl[quantization]

# Enable external plugins
ENV DOCLING_SERVE_ALLOW_EXTERNAL_PLUGINS=1

# Expose port
EXPOSE 5000

# Run server
CMD ["docling-serve", "run", "--host", "0.0.0.0", "--port", "5000"]
```

For GPU support with docker-compose:

```yaml
version: '3.8'
services:
  docling:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DOCLING_SERVE_ALLOW_EXTERNAL_PLUGINS=1
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Attention Backends

Qwen3-VL defaults to `flash_attention_2` for best performance. If `flash-attn` is not installed, the plugin automatically falls back to `eager` attention.

```python
# Default: flash_attention_2 (requires flash-attn package)
Qwen3VlOcrOptions(attn_implementation="flash_attention_2")

# Fallback: eager attention (no extra dependencies)
Qwen3VlOcrOptions(attn_implementation="eager")
```

To install flash-attn:
```bash
pip install flash-attn --no-build-isolation
```

## Multi-GPU Support

For systems with multiple GPUs, you can specify which GPU to use:

```python
# Use specific GPU
Qwen3VlOcrOptions(device="cuda:1")
```

Or use environment variable:
```bash
CUDA_VISIBLE_DEVICES=1 docling --allow-external-plugins --ocr-engine qwen3vl_ocr doc.pdf
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA OOM errors:

1. **Use quantization** (reduces VRAM from ~16GB to ~5GB):
   ```python
   Qwen3VlOcrOptions(quantization=Qwen3VlQuantization.INT4)
   ```

2. **Reduce `max_new_tokens`**:
   ```python
   Qwen3VlOcrOptions(max_new_tokens=2048)
   ```

3. **Reduce `page_scale`** (lower resolution OCR):
   ```python
   Qwen3VlOcrOptions(page_scale=1.5)
   ```

### flash-attn Not Available

If you see warnings about flash-attn, the plugin will fall back to eager attention automatically. To install flash-attn:

```bash
pip install flash-attn --no-build-isolation
```

### Plugin Not Found

Ensure both packages are in the same Python environment and external plugins are enabled:

```python
opts.allow_external_plugins = True  # Required!
```

### Slow First Inference

The first inference is slow because the model needs to be loaded. Subsequent inferences are much faster. The model is cached in memory after first load.

## Development

```bash
# Clone repository
git clone https://github.com/mayflower/docling-ocr-qwen3vl.git
cd docling-ocr-qwen3vl

# Install in development mode
pip install -e ".[test,quantization]"

# Run tests
pytest

# Run GPU test
python scripts/gpu_test.py
```

## License

MIT

## Acknowledgments

- [Qwen Team](https://github.com/QwenLM) for the Qwen3-VL model
- [Docling Project](https://github.com/docling-project) for the document processing framework
