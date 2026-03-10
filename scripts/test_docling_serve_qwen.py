#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.28",
#     "rich>=13",
# ]
# ///
"""Test docling-serve with full Qwen3-VL pipeline against a real PDF.

Sends a PDF URL to docling-serve (async) with Qwen3-VL OCR and polls for
completion. Displays the markdown result and any pipeline errors.

Usage:
    uv run scripts/test_docling_serve_qwen.py
    uv run scripts/test_docling_serve_qwen.py --base-url https://other-host:5001
    uv run scripts/test_docling_serve_qwen.py --timeout 600
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
    """Build JSON request body with full Qwen3-VL pipeline."""
    return {
        "sources": [{"kind": "http", "url": pdf_url}],
        "options": {
            "to_formats": ["md"],
            # OCR with Qwen3-VL
            "do_ocr": True,
            "force_ocr": True,
            "ocr_engine": "qwen3vl_ocr",
            # Table structure with Qwen3-VL
            "do_table_structure": True,
            "table_structure_custom_config": {"kind": "qwen3vl_table"},
            # Layout with Qwen3-VL
            "layout_custom_config": {"kind": "qwen3vl_layout"},
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
    console.print("[bold blue]Submitting conversion task...[/]")
    console.print(Syntax(json.dumps(body, indent=2), "json", theme="monokai", line_numbers=False))

    resp = client.post(f"{base_url}/v1/convert/source/async", json=body, timeout=30)
    if resp.status_code != 200:
        console.print(f"[bold red]Error {resp.status_code}:[/] {resp.text[:2000]}")
    resp.raise_for_status()

    data = resp.json()
    task_id = data["task_id"]
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
        pos = data.get("task_position")
        pos_str = f" (queue: {pos})" if pos is not None else ""

        console.print(f"  [{elapsed:6.1f}s] [bold]{status}[/]{pos_str}")

        if status == "success":
            return
        if status == "failure":
            console.print(f"[bold red]Task failed[/]")
            if data.get("error_message"):
                console.print(f"  Error: {data['error_message']}")
            sys.exit(1)

        time.sleep(5)


def fetch_result(task_id: str, base_url: str, client: httpx.Client) -> dict:
    resp = client.get(f"{base_url}/v1/result/{task_id}", timeout=120)
    resp.raise_for_status()
    console.print(f"[green]Result fetched[/] ({len(resp.content)} bytes)")
    return resp.json()


def print_result(result: dict) -> None:
    # Processing info
    proc_time = result.get("processing_time")
    if proc_time is not None:
        console.print(f"\n[bold]Processing time:[/] {proc_time:.1f}s")

    # Errors
    errors = result.get("errors", [])
    if errors:
        unique_msgs = {e.get("error_message", "?") for e in errors}
        console.print(f"\n[bold red]Pipeline errors ({len(errors)} total, {len(unique_msgs)} unique):[/]")
        for msg in sorted(unique_msgs):
            count = sum(1 for e in errors if e.get("error_message") == msg)
            console.print(f"  [{count}x] {msg}")

    # Document content
    doc = result.get("document", {})
    md = doc.get("md_content")

    if md:
        preview = md[:5000]
        if len(md) > 5000:
            preview += f"\n\n... ({len(md)} total chars)"
        console.print(Panel(preview, title=f"Markdown Output ({len(md)} chars)", border_style="green"))
    else:
        console.print("\n[bold yellow]No markdown content returned.[/]")
        if errors:
            console.print("[yellow]The pipeline errors above likely prevented content generation.[/]")

    # Save full result
    out_path = Path("scripts/test_output.json")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    console.print(f"[bold]Full result saved to[/] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test docling-serve with Qwen3-VL")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=600, help="Polling timeout (seconds)")
    parser.add_argument("--pdf-url", default=PDF_URL)
    args = parser.parse_args()

    console.print(Panel(
        f"[bold]docling-serve Qwen3-VL Integration Test[/]\n\n"
        f"Server:  {args.base_url}\n"
        f"PDF:     {args.pdf_url}\n"
        f"Timeout: {args.timeout}s\n\n"
        f"Features: Full Qwen3-VL pipeline (OCR, table, layout, pictures, code/formula)",
        border_style="blue",
    ))

    with httpx.Client(verify=False) as client:
        try:
            check_health(args.base_url, client)
        except httpx.HTTPError as exc:
            console.print(f"[bold red]Health check failed:[/] {exc}")
            sys.exit(1)

        t0 = time.monotonic()
        task_id = submit_task(args.pdf_url, args.base_url, client)

        console.print("[bold blue]Polling for completion...[/]")
        poll_task(task_id, args.base_url, client, args.timeout)

        result = fetch_result(task_id, args.base_url, client)
        total = time.monotonic() - t0
        console.print(f"[bold]Total wall time:[/] {total:.1f}s")

        print_result(result)

    console.print("\n[bold green]Done![/]")


if __name__ == "__main__":
    main()
