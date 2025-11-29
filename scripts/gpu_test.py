#!/usr/bin/env python3
"""GPU test that loads the model and runs inference on a PDF."""

import sys
from pathlib import Path


# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docling_ocr_qwen3vl.options import Qwen3VlOcrOptions, Qwen3VlPromptMode, Qwen3VlQuantization
from docling_ocr_qwen3vl.qwen_runner import Qwen3VlRunner


def main():
    print("Initializing Qwen3-VL runner with INT4 quantization...")
    options = Qwen3VlOcrOptions(
        device="cuda",
        max_new_tokens=4096,
        quantization=Qwen3VlQuantization.INT4,
    )
    runner = Qwen3VlRunner(options)

    print("Loading model (this may take a while)...")
    runner.ensure_loaded()
    print("Model loaded successfully!")

    # Load PDF and convert first page to image
    pdf_path = "/home/johann/scan.pdf"
    print(f"\nLoading PDF: {pdf_path}")

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[0]
    # Render at 150 DPI for good quality
    bitmap = page.render(scale=150 / 72)
    img = bitmap.to_pil()
    print(f"  Page size: {img.size}")

    print("\nRunning OCR inference...")
    result = runner.run(img, prompt_mode=Qwen3VlPromptMode.OCR)

    print("\n--- OCR Result ---")
    print(f"Full text:\n{result.text}")

    # Also test markdown mode
    print("\n--- Markdown Mode ---")
    result_md = runner.run(img, prompt_mode=Qwen3VlPromptMode.MARKDOWN)
    print(f"Markdown:\n{result_md.text}")

    # Test QwenVL HTML mode with bounding boxes
    print("\n--- QwenVL HTML Mode (with bounding boxes) ---")
    result_html = runner.run(img, prompt_mode=Qwen3VlPromptMode.QWENVL_HTML)
    print(f"Raw HTML:\n{result_html.raw_html[:500] if result_html.raw_html else 'None'}...")
    print(f"\nParsed {len(result_html.html_elements)} elements:")
    for el in result_html.html_elements[:10]:
        bbox_str = f"bbox={el.bbox}" if el.bbox else "no bbox"
        print(f"  [{el.element_type}] {el.text[:60]}... ({bbox_str})")

    print("\nGPU test passed!")


if __name__ == "__main__":
    main()
