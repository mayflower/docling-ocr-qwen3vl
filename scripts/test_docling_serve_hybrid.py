#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.28",
#     "rich>=13",
# ]
# ///
"""Test docling-serve with a hybrid pipeline.

Qwen3-VL for OCR (correct German diacritics) + docling's default
layout-heron and tableformer (better multi-column / table fidelity).
Saves to scripts/test_output_hybrid.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

PDF_URL = "https://mayflower.de/wp-content/uploads/2014/04/Security-Whitepaper-Conferencing.pdf"
DEFAULT_BASE_URL = "https://docling-serve.data.mayflower.zone"

console = Console()


def build_request_body(pdf_url: str) -> dict:
    return {
        "sources": [{"kind": "http", "url": pdf_url}],
        "options": {
            "to_formats": ["md"],
            "do_ocr": True,
            "force_ocr": True,
            "ocr_engine": "qwen3vl_ocr",  # Qwen3-VL OCR for clean German text
            "do_table_structure": True,    # default tableformer (no override)
            # No layout_custom_config -> uses default docling-layout-heron
            "do_picture_description": False,
            "do_code_enrichment": False,
            "do_formula_enrichment": False,
        },
    }


def check_health(base_url: str, client: httpx.Client) -> None:
    console.print(f"[bold blue]Checking health...[/] {base_url}/health")
    resp = client.get(f"{base_url}/health")
    resp.raise_for_status()
    console.print(f"[green]Health OK[/] {resp.json()}")


def submit_task(pdf_url: str, base_url: str, client: httpx.Client) -> str:
    body = build_request_body(pdf_url)
    console.print("[bold blue]Submitting (hybrid: qwen3vl OCR + default layout/table)...[/]")
    console.print(Syntax(json.dumps(body, indent=2), "json", theme="monokai", line_numbers=False))
    resp = client.post(f"{base_url}/v1/convert/source/async", json=body, timeout=30)
    if resp.status_code != 200:
        console.print(f"[bold red]Error {resp.status_code}:[/] {resp.text[:2000]}")
    resp.raise_for_status()
    task_id = resp.json()["task_id"]
    console.print(f"[green]Task submitted:[/] {task_id}")
    return task_id


def poll_task(task_id: str, base_url: str, client: httpx.Client, timeout: float) -> None:
    t0 = time.monotonic()
    url = f"{base_url}/v1/status/poll/{task_id}"
    while True:
        elapsed = time.monotonic() - t0
        if elapsed > timeout:
            console.print(f"[bold red]Timeout after {elapsed:.0f}s[/]")
            sys.exit(1)
        resp = client.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("task_status", "unknown")
        console.print(f"  [{elapsed:6.1f}s] [bold]{status}[/]")
        if status == "success":
            return
        if status == "failure":
            console.print(f"[bold red]Task failed[/] {data.get('error_message', '')}")
            sys.exit(1)
        time.sleep(5)


def fetch_result(task_id: str, base_url: str, client: httpx.Client) -> dict:
    resp = client.get(f"{base_url}/v1/result/{task_id}", timeout=120)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--pdf-url", default=PDF_URL)
    args = parser.parse_args()

    console.print(Panel(
        f"[bold]docling-serve HYBRID pipeline test[/]\n\n"
        f"OCR:     qwen3vl_ocr (Qwen3-VL)\n"
        f"Layout:  docling-layout-heron (default)\n"
        f"Tables:  docling_tableformer (default)\n\n"
        f"Server:  {args.base_url}\n"
        f"PDF:     {args.pdf_url}\n"
        f"Timeout: {args.timeout}s",
        border_style="magenta",
    ))

    with httpx.Client(verify=False) as client:
        check_health(args.base_url, client)
        t0 = time.monotonic()
        task_id = submit_task(args.pdf_url, args.base_url, client)
        console.print("[bold blue]Polling...[/]")
        poll_task(task_id, args.base_url, client, args.timeout)
        result = fetch_result(task_id, args.base_url, client)
        total = time.monotonic() - t0
        console.print(f"[bold]Total wall time:[/] {total:.1f}s")

    out_path = Path("scripts/test_output_hybrid.json")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    console.print(f"[bold]Saved:[/] {out_path}")

    proc_time = result.get("processing_time")
    if proc_time is not None:
        console.print(f"Processing time: {proc_time:.1f}s")

    errors = result.get("errors", [])
    if errors:
        console.print(f"[yellow]{len(errors)} pipeline errors[/]")

    md = result.get("document", {}).get("md_content", "")
    if md:
        console.print(Panel(md[:2000] + ("\n...\n" if len(md) > 2000 else ""),
                            title=f"Markdown ({len(md)} chars)", border_style="green"))


if __name__ == "__main__":
    main()
