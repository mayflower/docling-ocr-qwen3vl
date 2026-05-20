# Recommended pipeline recipe

This document captures the configuration we recommend in practice after measuring
quality and speed on real bilingual (German + English) PDFs with mixed
multi-column layouts. If you only read one section: jump to
[Recommended request](#recommended-request).

The short version: **use Qwen3-VL only for the stages where it actually wins
(OCR text fidelity, picture description), and let docling's dedicated models
handle layout and table structure** — they preserve column / table structure
better and run an order of magnitude faster.

---

## Recommended request

`POST /v1/convert/source/async` against a docling-serve instance running the
mayflower-qwen3vl image:

```jsonc
{
  "sources": [{"kind": "http", "url": "https://your.host/your.pdf"}],
  "options": {
    "to_formats": ["md"],
    "do_ocr": true,
    "force_ocr": true,
    "ocr_engine": "qwen3vl_ocr",
    "do_table_structure": true,
    "do_picture_description": true,
    "picture_description_custom_config": {"kind": "qwen3vl"},
    "do_code_enrichment": false,
    "do_formula_enrichment": false
  }
}
```

Stage-by-stage breakdown:

| Stage | Engine | Why |
|---|---|---|
| OCR | `qwen3vl_ocr` | Correct German diacritics (`ü ö ä ß`); preserves word spacing; handles mixed-language content. RapidOCR mangles these. |
| Layout | default (`docling-layout-heron`) | Catches richer heading hierarchy and detects 2/3/4-column regions as tables. Qwen3-VL's layout call tends to linearize columns into one paragraph. |
| Table structure | default (`docling_tableformer`) | Builds proper markdown tables from the multi-column blocks. Qwen3-VL produces 0 tables on the same input. |
| Picture description | `qwen3vl` (custom_config kind dispatch) | Detailed multi-sentence captions; **reuses the model already in VRAM** for OCR, so almost no extra wall-clock cost. |
| Code / formula enrichment | off | Add as needed; the plugin supports them but they trigger extra inference passes. |

---

## Why this combination

Measured on a 10-page mixed German/English security whitepaper, RTX 3080 Ti
Laptop (16 GB VRAM):

| Pipeline | Speed | German diacritics | Column→table | Tables detected |
|---|---|---|---|---|
| All Qwen3-VL | 638 s | ✅ perfect | ❌ flattens | 0 |
| All default | 10 s | ❌ broken (`möglich`→`moglich`, `über`→`iber`) | ✅ tables | 16 |
| **Hybrid (recommended)** | **305 s** | **✅ perfect** | **✅ tables** | **9** |
| Hybrid + Qwen3-VL pic desc | 330 s | ✅ | ✅ | 9 |

The hybrid + picture-description run beats all-Qwen3-VL on speed by ~2×, and
beats all-default on text quality on every German word with an umlaut. The
picture-description tax is only +25 s for 28 images because the descriptions
reuse the already-loaded Qwen3-VL model.

---

## How the picture-description dispatch works

This is the non-obvious bit. Upstream docling-serve's new picture description
API (`PictureDescriptionVlmEngineOptions` with `model_spec` / `engine_options`)
doesn't route to plugin-registered options classes — it always coerces the
request dict into its own engine schema, which would force a second model load
and runs into transformers 4.57.x quirks with bnb-4bit configs.

The mayflower fork of docling-jobkit adds an escape hatch in
`DoclingConverterManager._parse_picture_description_options`: a custom_config
dict whose `kind` matches a plugin-registered picture description factory
kind is dispatched to the plugin's own options class. The factory then
dispatches by kind to `Qwen3VlPictureDescriptionModel`, which uses the plugin's
shared model singleton (the same one already serving OCR / layout / table
inference). Net effect: **one Qwen3-VL model in VRAM for all stages**, no
double-loading, no transformers compatibility patches needed at runtime.

The plugin still ships defensive `to_dict` / `to_diff_dict` patches and a
`qwen3vl` preset registration on `PictureDescriptionVlmEngineOptions`. Those
cover the preset-based path
(`picture_description_preset: "qwen3vl"`) for completeness, but that path
loads a second model copy through docling's transformers VLM engine. Prefer
the custom_config kind dispatch shown above.

---

## Server-side requirements

### Image

`ghcr.io/mayflower/docling-serve:latest` (built from `mayflower/docling-serve`
main via [`Containerfile.mayflower-qwen3vl`][cf]) bakes in:

- `mayflower/docling-jobkit` (the fork with the kind escape hatch)
- `docling-ocr-qwen3vl` (this plugin, including preset registration and
  defensive transformers patches)
- `DOCLING_SERVE_ALLOW_EXTERNAL_PLUGINS=true`
- `DOCLING_SERVE_ALLOW_CUSTOM_OCR_CONFIG=true`
- `DOCLING_SERVE_ALLOW_CUSTOM_TABLE_STRUCTURE_CONFIG=true`
- `DOCLING_SERVE_ALLOW_CUSTOM_LAYOUT_CONFIG=true`
- `DOCLING_SERVE_ALLOW_CUSTOM_PICTURE_DESCRIPTION_CONFIG=true`

A build-time assertion verifies the docling-jobkit escape hatch is actually
present in the installed manager.py, so a future drift sync that drops the
fix fails the build instead of silently regressing to upstream behaviour.

[cf]: https://github.com/mayflower/docling-serve/blob/main/Containerfile.mayflower-qwen3vl

### Default model

The plugin loads `unsloth/Qwen3-VL-4B-Instruct-bnb-4bit` (~5 GB int4) as a
shared singleton across all stages. With OCR + layout + table + picture
description active you need roughly 6–8 GB VRAM during inference; 16 GB is
comfortable. Override with `model_repo_id` in OCR options if you want the
official Qwen3-VL bf16 model (~8 GB) on a larger GPU.

### Transformers version

The plugin pins `transformers>=4.55,<5`. The unsloth bnb-4bit model's embedded
`quantization_config` doesn't parse cleanly in transformers 5.x (FP4 init
assertion in bitsandbytes 0.49). The image installs the right version
automatically; if you bring your own venv, install with `pip install
docling-ocr-qwen3vl` (don't bypass the pin).

---

## End-to-end example

```bash
curl -X POST http://docling-serve.your-host/v1/convert/source/async \
  -H "Content-Type: application/json" \
  -d '{
    "sources": [{"kind": "http", "url": "https://example.com/doc.pdf"}],
    "options": {
      "to_formats": ["md"],
      "do_ocr": true,
      "force_ocr": true,
      "ocr_engine": "qwen3vl_ocr",
      "do_table_structure": true,
      "do_picture_description": true,
      "picture_description_custom_config": {"kind": "qwen3vl"}
    }
  }'
# → { "task_id": "...", "task_status": "pending" }

TASK_ID=...
# Poll until success
curl -s http://docling-serve.your-host/v1/status/poll/$TASK_ID
# Fetch result
curl -s http://docling-serve.your-host/v1/result/$TASK_ID
```

For a ready-to-run client see [`scripts/test_docling_serve_hybrid_pic.py`][hp]
(`uv run scripts/test_docling_serve_hybrid_pic.py --base-url
http://localhost:5001`). Companion scripts `test_docling_serve_hybrid.py` (no
pic desc) and `test_docling_serve_default.py` (no plugin at all) are useful
for A/B-ing the pipeline.

[hp]: scripts/test_docling_serve_hybrid_pic.py

---

## When to use other configurations

- **All-default pipeline** (no plugin): great if your documents are
  English-only and you don't need image captions. ~60× faster than any
  Qwen3-VL pipeline.
- **All Qwen3-VL** (`ocr_engine: qwen3vl_ocr` +
  `table_structure_custom_config: {"kind": "qwen3vl_table"}` +
  `layout_custom_config: {"kind": "qwen3vl_layout"}`): use only if you
  specifically need Qwen3-VL's layout reasoning, accept that multi-column
  content gets linearized, and have the wall-clock budget.
- **Picture description only with default OCR**: skip `ocr_engine` and the
  table/layout custom configs but keep `picture_description_custom_config:
  {"kind": "qwen3vl"}`. Useful when you don't need German OCR fidelity but do
  want detailed image captions.

---

## Known limitations

- **Multi-column non-tabular flows are still hard.** The default tableformer
  catches grid-like multi-column blocks well, but irregular 4-column
  "narrative" pages (e.g. DSGVO/Datenschutz cards laid out in 4 vertical
  blocks of prose) collapse into one paragraph. There's no good fix today.
- **The bnb-4bit model + transformers 5.x is broken** at multiple layers
  (config repr, config dict for `pre_quantized` detection). Keep
  `transformers<5` until upstream resolves this, or use the official
  `Qwen/Qwen3-VL-4B-Instruct` bf16 model (needs more VRAM).
- **The `picture_description_preset: "qwen3vl"` path works but loads a second
  model copy** through docling's transformers VLM engine. Use the
  custom_config kind dispatch (`picture_description_custom_config:
  {"kind": "qwen3vl"}`) instead.
- **`do_code_enrichment` / `do_formula_enrichment`** with Qwen3-VL aren't
  benchmarked in the recipe; defaults are off.
