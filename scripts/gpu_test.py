#!/usr/bin/env python3
"""GPU test that loads the model and runs inference on a test image."""

import sys
from pathlib import Path


# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docling_ocr_qwen3vl.options import Qwen3VlOcrOptions, Qwen3VlPromptMode
from docling_ocr_qwen3vl.qwen_runner import Qwen3VlRunner
from PIL import Image


def main():
    print("Initializing Qwen3-VL runner...")
    options = Qwen3VlOcrOptions(
        device="cuda",
        max_new_tokens=1024,
    )
    runner = Qwen3VlRunner(options)

    print("Loading model (this may take a while)...")
    runner.ensure_loaded()
    print("Model loaded successfully!")

    # Create a simple test image with text
    print("\nCreating test image...")
    img = Image.new("RGB", (400, 200), color="white")

    # If PIL has ImageDraw, add some text
    try:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        draw.text((20, 20), "Hello World!", fill="black")
        draw.text((20, 60), "This is a test.", fill="black")
        draw.text((20, 100), "Qwen3-VL OCR Plugin", fill="black")
    except ImportError:
        print("  ImageDraw not available, using blank image")

    print("Running OCR inference...")
    result = runner.run(img, prompt_mode=Qwen3VlPromptMode.OCR)

    print("\n--- OCR Result ---")
    print(f"Full text:\n{result.text}")
    print(f"\nParagraphs ({len(result.paragraphs)}):")
    for i, p in enumerate(result.paragraphs):
        print(f"  {i + 1}. {p}")

    print("\nGPU test passed!")


if __name__ == "__main__":
    main()
