#!/usr/bin/env python3
"""Smoke test that validates plugin structure without loading the model."""

import sys
from pathlib import Path


# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docling_ocr_qwen3vl import Qwen3VlOcrModel, Qwen3VlOcrOptions, Qwen3VlPromptMode
from docling_ocr_qwen3vl.plugins import ocr_engines
from docling_ocr_qwen3vl.prompts import resolve_prompt


def main():
    print("Testing plugin registration...")
    engines = ocr_engines()
    assert "ocr_engines" in engines
    assert Qwen3VlOcrModel in engines["ocr_engines"]
    print("  Plugin registration OK")

    print("Testing options...")
    opts = Qwen3VlOcrOptions()
    assert opts.kind == "qwen3vl_ocr"
    assert opts.model_repo_id == "Qwen/Qwen3-VL-8B-Thinking"
    print(f"  Options OK: {opts.kind}")

    print("Testing prompts...")
    for mode in Qwen3VlPromptMode:
        prompt = resolve_prompt(mode, {})
        assert prompt, f"Empty prompt for {mode}"
        print(f"  {mode.value}: {prompt[:50]}...")

    print("Testing prompt overrides...")
    custom = "Custom prompt"
    result = resolve_prompt(Qwen3VlPromptMode.OCR, {Qwen3VlPromptMode.OCR.value: custom})
    assert result == custom
    print("  Prompt overrides OK")

    print("\nAll smoke tests passed!")


if __name__ == "__main__":
    main()
