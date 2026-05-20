# Picture description comparison: Qwen3-VL vs SmolVLM

Apples-to-apples benchmark of the two picture-description options available
in the mayflower docling-serve image. Both runs use the same OCR
(`qwen3vl_ocr`), the same default `docling-layout-heron` + `docling_tableformer`
for layout and tables, and the same input document. Only the picture
description engine is swapped.

## Setup

- **Server**: local docling-serve dev, mayflower fork, with
  `picture_description_custom_config={"kind":"qwen3vl"}` bridge installed
  (mayflower/docling-jobkit's kind escape hatch).
- **Hardware**: NVIDIA GeForce RTX 3080 Ti Laptop, 16 GB VRAM.
- **Input**: 10-page bilingual (German + English) Mayflower Security
  Whitepaper PDF, multi-column layout, mixed photos / logos / icons.
- **Pipeline configuration**, identical across both runs:
  - `ocr_engine: "qwen3vl_ocr"` (Qwen3-VL, unsloth bnb-4bit)
  - default `docling-layout-heron` for layout
  - default `docling_tableformer` for tables
  - `do_picture_description: true`
- **Variable**: picture description engine
  - Run A: `picture_description_custom_config: {"kind": "qwen3vl"}` —
    factory-dispatched into the plugin's `Qwen3VlPictureDescriptionModel`,
    which reuses the shared Qwen3-VL singleton already serving OCR.
  - Run B: `picture_description_preset: "smolvlm"` — docling's upstream
    `PictureDescriptionVlmEngineModel` with SmolVLM-256M-Instruct loaded as
    a separate model in VRAM.
- **PDF**: 10 pages, 28 image regions detected, of which 2 cleared the
  default `picture_area_threshold: 0.05` and got captioned.

Raw artifacts:
- `scripts/test_output_hybrid_pic.json` — Run A (qwen3vl)
- `scripts/test_output_hybrid_smolvlm.json` — Run B (smolvlm)

## Results

| Metric | Qwen3-VL pic desc (A) | SmolVLM pic desc (B) |
|---|---|---|
| Processing time | 321 s | 304 s |
| Pipeline errors | 0 | 0 |
| Pictures captioned | 2 of 28 | 2 of 28 |
| Caption quality | Detailed, multi-sentence | Generic one-liners |
| Output cleanliness | Clean | Broken `<end_of_utteranc` truncation |
| Extra VRAM cost | ~0 (singleton shared with OCR) | +500 MB (separate model load) |

SmolVLM is 17 s (5%) faster. The win is rounding error in a 5-minute pipeline
dominated by Qwen3-VL OCR. SmolVLM does not earn its separate model load at
this output quality.

## Caption samples

Same image, both engines. Image 1 — BITS & PRETZELS hero / case-study slide:

> **Qwen3-VL**
> This is a promotional or informational graphic, likely for a corporate or
> marketing purpose, designed to highlight a video platform called
> "BITS & PRETZELS." The image is split into two distinct vertical sections.

> **SmolVLM**
> In this image, we can see a group of people. There is a blue
> background.\<end\_of\_utteranc

Image 2 — conference hall photo with framed video feed overlay:

> **Qwen3-VL**
> This is a promotional graphic or advertisement for digital conferencing,
> designed to convey the concept of virtual connection and collaboration.
> The image is split into two main visual areas: a dark, blurred background
> on the left and a clear, framed video feed on the right.

> **SmolVLM**
> In this image we can see a group of people standing and watching something.
> In the background there is a blur image.\<end\_of\_utteranc

Qwen3-VL reads on-image text ("BITS & PRETZELS"), identifies purpose
("advertisement for digital conferencing"), and describes composition
("split into two main visual areas: a dark, blurred background on the left
and a clear, framed video feed on the right"). SmolVLM produces generic
"group of people" descriptions and ships broken EOS tokens that leak into
the markdown output (the `<end_of_utterance>` marker hits the
`max_new_tokens=200` cap mid-token).

## Recommendation

For German technical / marketing PDFs with embedded photos that contain
on-image text or convey context: **use Qwen3-VL via the custom_config kind
dispatch**. Speed cost is negligible — the singleton already in memory for
OCR is reused — and caption quality is qualitatively better in every
dimension that matters (content, on-image text, intent, composition).

SmolVLM remains the right pick only when:

- OCR is also a non-Qwen engine (so no shared singleton),
- The deployment is VRAM-constrained below ~6 GB,
- And caption fidelity is not a requirement.

For everything else in this stack, the recommended request body in
[`RECIPES.md`](RECIPES.md) (qwen3vl OCR + default layout/tables +
`picture_description_custom_config: {"kind": "qwen3vl"}`) stays the default.
