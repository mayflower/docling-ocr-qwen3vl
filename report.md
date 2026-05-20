# Picture description comparison: Qwen3-VL vs SmolVLM vs Granite Vision

Benchmark of the three picture-description engines available in the mayflower
docling-serve image, run against the same input document.

## Setup

- **Server**: local docling-serve dev, mayflower fork, with the
  `picture_description_custom_config={"kind":"qwen3vl"}` bridge installed
  (mayflower/docling-jobkit's kind escape hatch).
- **Hardware**: NVIDIA GeForce RTX 3080 Ti Laptop, 16 GB VRAM.
- **Input**: 10-page bilingual (German + English) Mayflower Security
  Whitepaper PDF, multi-column layout, mixed photos / logos / icons.
- **Picture area threshold**: default `0.05`, so only photo-like regions get
  captioned. Both photos in the document cleared the threshold; logos and
  small icons did not.

Raw artifacts (in `scripts/`):
- `test_output_hybrid_pic.json` — Qwen3-VL pic desc
- `test_output_hybrid_smolvlm.json` — SmolVLM pic desc
- `test_output_hybrid_granite.json` — Granite Vision pic desc

## VRAM situation

This benchmark exists because the docling picture-description architecture
(new `PictureDescriptionVlmEngineModel` + `TransformersVlmEngine`) loads its
model independently of the OCR stage. Picking a non-Qwen picture description
engine means **two VLMs in VRAM at once** — the OCR model and the picture
description model — for the entire conversion. This matters on a 16 GB GPU:

| Picture description model | + Qwen3-VL OCR singleton fits? |
|---|---|
| Qwen3-VL (via kind dispatch) | **shared singleton — only one model in VRAM** |
| SmolVLM-256M (~500 MB) | yes |
| Granite Vision 3.3-2B (~5 GB bf16) | **no — OOMs at inference** |

For Granite Vision, this means we couldn't run apples-to-apples against the
other two. The Granite run uses **default OCR (RapidOCR)** instead of
Qwen3-VL OCR, which keeps only Granite in VRAM.

## Results

| Pipeline | OCR engine | Pic desc engine | Proc time | Pic capt | Errors |
|---|---|---|---:|---:|---:|
| **Qwen3-VL bridge** | Qwen3-VL | Qwen3-VL (singleton) | 321 s | 2 of 28 | 0 |
| Hybrid + SmolVLM | Qwen3-VL | SmolVLM preset | 304 s | 2 of 28 | 0 |
| Default + Granite | RapidOCR | Granite Vision preset | 21 s | 2 of 28 | 0 |

The Granite row is **not** time-comparable to the other two — almost all the
21 s saving vs Qwen3-VL OCR is the OCR engine difference, not the picture
description engine.

## Caption quality, side by side

Same two photo images across all three runs.

### Image 1 — BITS & PRETZELS hero / case-study slide

> **Qwen3-VL**
> This is a promotional or informational graphic, likely for a corporate or
> marketing purpose, designed to highlight a video platform called
> "BITS & PRETZELS." The image is split into two distinct vertical sections.

> **Granite Vision**
> a screen with the words bits & pretzels on it and a crowd of people in
> front of i *(truncated at max_new_tokens)*

> **SmolVLM**
> In this image, we can see a group of people. There is a blue
> background.\<end\_of\_utteranc

### Image 2 — conference hall photo with framed video feed overlay

> **Qwen3-VL**
> This is a promotional graphic or advertisement for digital conferencing,
> designed to convey the concept of virtual connection and collaboration.
> The image is split into two main visual areas: a dark, blurred background
> on the left and a clear, framed video feed on the right.

> **Granite Vision**
> digital conference with the text digital conference on it and a large
> screen showing a crowd of people sitting in chairs and looking at a screen.

> **SmolVLM**
> In this image we can see a group of people standing and watching
> something. In the background there is a blur image.\<end\_of\_utteranc

### Observations

- **Qwen3-VL** reads on-image text (`BITS & PRETZELS`), names the image
  *purpose* ("advertisement for digital conferencing"), and describes the
  *composition* ("split into two main visual areas: a dark, blurred
  background on the left and a clear, framed video feed on the right").
  Multi-sentence, clean.
- **Granite Vision** reads on-image text but in a CLIP-like format
  ("a screen with the words X"). Single sentence, sometimes truncated at
  the `max_new_tokens` cap, and shows verbatim repetition ("digital
  conference … digital conference"). Better than SmolVLM, well below
  Qwen3-VL.
- **SmolVLM** misses every piece of on-image text, gives generic "group of
  people / blue background" descriptions, and ships broken
  `<end_of_utterance>` markers when its EOS token gets cut off by
  `max_new_tokens=200`.

## German text quality (separate axis)

Because the Granite run forced us off Qwen3-VL OCR, its document body shows
the usual RapidOCR weaknesses on German:

- `Lösung` → `Losung`
- `München` → `Munchen`
- `möglich` → `moglich`
- Word-spacing errors: `Vorraussetzungenerfillen`, `DieZugriffsschlussel`
- Digits / letters confused: `5.000` → `5.0o0`

Qwen3-VL OCR keeps all of these correct (umlauts, ß, word spacing). For
German technical / marketing PDFs that's the difference between searchable
content and lossy noise — independent of which picture description engine
you choose.

## Recommendation

- **For Qwen3-VL OCR (recommended for German content)**:
  picture description should also be Qwen3-VL via the kind dispatch
  (`picture_description_custom_config: {"kind": "qwen3vl"}`). One model in
  VRAM, no compatibility issues, best caption quality.
- **If you must use RapidOCR / a non-Qwen OCR**: Granite Vision is the
  better picture description option of the two non-Qwen choices — it at
  least reads on-image text. SmolVLM produces captions that aren't worth
  the model load on this kind of content.
- **SmolVLM** earns its slot only on tightly VRAM-constrained deployments
  where caption *coverage* matters more than caption *quality*, and where
  on-image text reading is not required.

The default request body in [`RECIPES.md`](RECIPES.md) (qwen3vl OCR +
default layout/tables + `picture_description_custom_config: {"kind":
"qwen3vl"}`) stays the default for this stack.
