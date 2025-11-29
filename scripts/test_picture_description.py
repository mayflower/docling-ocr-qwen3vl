#!/usr/bin/env python3
"""Test script for Qwen3-VL picture description model."""

import sys
from pathlib import Path


# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling_ocr_qwen3vl.options import Qwen3VlPictureDescriptionOptions, Qwen3VlQuantization
from docling_ocr_qwen3vl.picture_description_model import Qwen3VlPictureDescriptionModel


def main():
    print("=== Qwen3-VL Picture Description Test ===\n")

    # Create options with 4-bit quantization to reduce VRAM
    options = Qwen3VlPictureDescriptionOptions(
        quantization=Qwen3VlQuantization.INT4,
        max_new_tokens=512,
    )

    print(f"Model: {options.model_repo_id}")
    print(f"Quantization: {options.quantization.value}")
    print(f"Max tokens: {options.max_new_tokens}")
    print()

    # Initialize the model
    print("Initializing Qwen3-VL picture description model...")
    model = Qwen3VlPictureDescriptionModel(
        enabled=True,
        enable_remote_services=False,
        artifacts_path=None,
        options=options,
        accelerator_options=AcceleratorOptions(),
    )
    print("Model loaded successfully!")
    print(f"Provenance: {model.provenance}")
    print()

    # Load a test image - extract from PDF first page
    pdf_path = "/home/johann/scan.pdf"
    print(f"Loading test image from PDF: {pdf_path}")

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[0]
    # Render at 150 DPI
    bitmap = page.render(scale=150 / 72)
    img = bitmap.to_pil()
    print(f"Image size: {img.size}")
    print()

    # Run picture description
    print("Running picture description inference...")
    descriptions = list(model._annotate_images([img]))

    print("\n=== Picture Description Result ===")
    for i, desc in enumerate(descriptions):
        print(f"\nImage {i + 1}:")
        print("-" * 40)
        print(desc)
        print("-" * 40)

    print("\n=== SUCCESS: Picture description test passed! ===")


if __name__ == "__main__":
    main()
