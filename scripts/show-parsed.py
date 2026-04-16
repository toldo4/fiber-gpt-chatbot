#!/usr/bin/env python3
"""
show-parsed.py — Display a summary of parsed-docs.json.

Usage:
  python scripts/show-parsed.py [--file FILE] [--section SECTION] [--text]
"""
import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

INPUT = Path(__file__).parent / "parsed-docs.json"


def main():
    parser = argparse.ArgumentParser(description="Show parsed files and sections from parsed-docs.json")
    parser.add_argument("--file", "-f", help="Filter by filename (partial match)")
    parser.add_argument("--section", "-s", help="Filter by section name (partial match)")
    parser.add_argument("--text", "-t", action="store_true", help="Show text content (truncated)")
    parser.add_argument("--full", action="store_true", help="Show full text content (no truncation)")
    args = parser.parse_args()

    if not INPUT.exists():
        print(f"ERROR: {INPUT} not found. Run docling-parse.py first.", file=sys.stderr)
        sys.exit(1)

    records = json.loads(INPUT.read_text(encoding="utf-8"))
    if not records:
        print("No records found in parsed-docs.json.")
        return

    # Apply filters
    if args.file:
        records = [r for r in records if args.file.lower() in r["file"].lower()]
    if args.section:
        records = [r for r in records if args.section.lower() in r["section"].lower()]

    if not records:
        print("No records match the given filters.")
        return

    # Group: file → sections → blocks
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        grouped[r["file"]][r["section"]].append(r["text"])

    total_files    = len(grouped)
    total_sections = sum(len(secs) for secs in grouped.values())
    total_blocks   = len(records)

    print(f"\n{'='*60}")
    print(f"  parsed-docs.json — {total_files} file(s), {total_sections} section(s), {total_blocks} block(s)")
    print(f"{'='*60}\n")

    for filename, sections in sorted(grouped.items()):
        file_blocks = sum(len(b) for b in sections.values())
        print(f"FILE: {filename}  ({len(sections)} sections, {file_blocks} blocks)")
        print(f"{'─'*60}")

        for section, blocks in sections.items():
            print(f"  SECTION [{len(blocks)} block(s)]: {section}")

            if args.text or args.full:
                for i, block in enumerate(blocks, 1):
                    display = block if args.full else (block[:200] + "…" if len(block) > 200 else block)
                    print(f"    [{i}] {display}")

        print()

    print(f"Total: {total_files} file(s) · {total_sections} section(s) · {total_blocks} text block(s)")


if __name__ == "__main__":
    main()
