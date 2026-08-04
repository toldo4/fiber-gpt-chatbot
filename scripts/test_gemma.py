"""
Drive the AOAC decision tree with Gemma for extraction and the local corpus for
evidence.

The agent works the tree one node at a time. Product facts it infers from the
description with retrieval support; anything only the customer can know — their
labelling obligation, their jurisdiction's fibre definition — it stops and asks.

Usage:
    python3 test_gemma.py
    python3 test_gemma.py "whole wheat bread enriched with inulin (FOS), EU market"

    # answer up front instead of being prompted
    python3 test_gemma.py --answer needs_idf_sdf_separate=yes "canned kidney beans"

    # never prompt; unanswered questions come back as `unresolved`
    python3 test_gemma.py --non-interactive "canned kidney beans"

    --anthropic     use Claude instead of Gemma for extraction
    --no-rag        skip retrieval (useful for isolating retrieval problems)
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from decision_tree_engine import DecisionTree, DecisionTreeEngine, Session, TraversalResult
from extractors import AnthropicExtractor, Extraction, ExtractionError, GemmaExtractor
from rag_index import BoundRetriever, HybridIndex, OpenAIEmbedder

ROOT = Path(__file__).parent.parent
HERE = Path(__file__).parent
ENV = {**dotenv_values(ROOT / ".env.local"), **dotenv_values(HERE / ".env")}

GEMMA_MODEL = "gemma-4-26b-a4b-it"
ANTHROPIC_MODEL = "claude-sonnet-5"
EMBED_MODEL = "text-embedding-3-small"      # must match the model the index was built with

SAMPLE_QUERY = (
    "canned kidney beans, requires heating before eating, "
    "no added NDO, high resistant starch content"
)

RULE = "─" * 72


# ── Presentation ──────────────────────────────────────────────────────────────

def heading(title: str) -> None:
    print(f"\n{RULE}\n  {title}\n{RULE}")


def wrap(text: str, indent: str = "    ", label: str = "") -> str:
    return textwrap.fill(
        " ".join(text.split()),
        width=88,
        initial_indent=indent + label,
        subsequent_indent=indent + " " * len(label),
    )


class Recorder:
    """Prints the agent's work as it happens and keeps a structured log."""

    def __init__(self, verbose: bool = True) -> None:
        self.entries: list[dict] = []
        self.verbose = verbose

    def log(self, entry: dict) -> None:
        self.entries.append({**entry, "ts": datetime.now().isoformat(timespec="seconds")})

    def __call__(self, kind: str, payload: dict) -> None:
        if kind == "retrieval":
            hits = payload["hits"]
            if self.verbose:
                print(f"\n  retrieval for {payload['attribute']}")
                print(wrap(payload["query"], "      ", "query: "))
                for i, hit in enumerate(hits, 1):
                    flags = []
                    if hit.dense_rank is not None:
                        flags.append(f"dense#{hit.dense_rank + 1}")
                    if hit.lexical_rank is not None:
                        flags.append(f"bm25#{hit.lexical_rank + 1}")
                    print(f"      [{i}] {hit.cite()[:60]:60s} {'+'.join(flags)}")
            self.log({
                "type": "retrieval",
                "attribute": payload["attribute"],
                "query": payload["query"],
                "hits": [
                    {"file": h.file, "section": h.section, "score": round(h.score, 5),
                     "dense_rank": h.dense_rank, "lexical_rank": h.lexical_rank}
                    for h in hits
                ],
            })

        elif kind == "extraction":
            extraction: Extraction = payload["extraction"]
            if self.verbose:
                print(f"      → {payload['attribute']} = {extraction.value!r} "
                      f"(confidence {extraction.confidence:.2f})")
                if extraction.reasoning:
                    print(wrap(extraction.reasoning, "        "))
                for ev in extraction.evidence:
                    print(f"        cites {ev}")
            self.log({
                "type": "extraction",
                "attribute": payload["attribute"],
                "value": extraction.value,
                "confidence": extraction.confidence,
                "reasoning": extraction.reasoning,
                "evidence": [asdict(e) for e in extraction.evidence],
            })

        elif kind == "answer":
            self.log({"type": "answer", **payload})

    def save(self) -> Path:
        path = HERE / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(path, "w") as f:
            for entry in self.entries:
                f.write(json.dumps(entry, default=str) + "\n")
        return path


def print_result(result: TraversalResult, tree: DecisionTree) -> None:
    heading("RECOMMENDATION")
    print(f"  Method     : {result.method}")
    print(f"  Confidence : {result.confidence:.2f}"
          f"{'  (every answer supported)' if result.is_reliable else ''}")

    if result.warning:
        print("\n  ⚠ Warning")
        print(wrap(result.warning_message, "    "))
        if result.can_override:
            print("    (re-run with --override to continue past this stop)")

    if result.unresolved:
        print("\n  ⚠ Answered without these — they could change the recommendation:")
        for key in result.unresolved:
            spec = tree.attributes[key]
            print(wrap(spec.ask, "    ", "· "))
            print(wrap(spec.why, "        "))

    if result.assumptions:
        print("\n  Assumptions made:")
        for assumption in result.assumptions:
            print(wrap(f"· {assumption}", "    "))

    heading("HOW IT GOT THERE")
    for i, step in enumerate(result.steps, 1):
        extraction = step.extraction
        origin = extraction.origin if extraction else "—"
        confidence = f"{extraction.confidence:.2f}" if extraction else "—"
        print(f"\n  {i}. {step.question}")
        print(f"     answer: {step.value!r}   [{origin}, confidence {confidence}]")
        print(f"     → {step.edge.result or step.edge.next}")
        if extraction and extraction.reasoning and origin == "model":
            print(wrap(extraction.reasoning, "        "))

    evidence = result.evidence()
    if evidence:
        heading("EVIDENCE CITED")
        for ev in evidence:
            print(f"\n  {ev}")
            print(wrap(ev.quote, "      "))


# ── Interaction ───────────────────────────────────────────────────────────────

def ask_user(session: Session) -> None:
    """Prompt for whatever the agent could not establish on its own."""
    while session.pending:
        question = session.pending
        heading("QUESTION FOR YOU")
        print(wrap(question.prompt, "  "))
        print(wrap(f"Why it matters: {question.why}", "    "))

        weak = question.because_low_confidence
        if weak is not None:
            print(f"\n    (the model guessed {weak.value!r} but was only "
                  f"{weak.confidence:.0%} confident)")

        options = "/".join(question.options) if question.options else "a number, or 'unknown'"
        try:
            answer = input(f"\n  [{options}] > ").strip()
        except EOFError:
            print("\n  no input available — recording as unknown")
            answer = "unknown"
        session.answer(answer or "unknown")


# ── Main ──────────────────────────────────────────────────────────────────────

def build_engine(args, recorder: Recorder) -> DecisionTreeEngine:
    tree = DecisionTree.from_yaml(HERE / "decision_tree.yaml")

    retriever = None
    if not args.no_rag:
        openai_key = ENV.get("OPENAI_API_KEY", "")
        if not openai_key:
            raise SystemExit("OPENAI_API_KEY is required for retrieval (or pass --no-rag)")
        index = HybridIndex.load(ROOT / "index-parts")
        print(f"  corpus: {len(index.chunks)} usable chunks")
        retriever = BoundRetriever(index, OpenAIEmbedder(openai_key, model=EMBED_MODEL))

    if args.anthropic:
        import anthropic

        key = ENV.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise SystemExit("ANTHROPIC_API_KEY is not set in .env.local")
        extractor: Any = AnthropicExtractor(anthropic.Anthropic(api_key=key), ANTHROPIC_MODEL)
        print(f"  extractor: {ANTHROPIC_MODEL}")
    else:
        key = ENV.get("GOOGLE_GENERATIVE_AI_API_KEY", "")
        if not key:
            raise SystemExit("GOOGLE_GENERATIVE_AI_API_KEY is not set")
        extractor = GemmaExtractor(
            key,
            GEMMA_MODEL,
            on_retry=lambda n, total, wait, err: print(
                f"      [retry {n}/{total}] {type(err).__name__} — waiting {wait}s"
            ),
        )
        print(f"  extractor: {GEMMA_MODEL}")

    return DecisionTreeEngine(tree, extractor, retriever=retriever, on_event=recorder)


def parse_answers(pairs: list[str]) -> dict[str, str]:
    answers: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--answer expects attribute=value, got {pair!r}")
        key, value = pair.split("=", 1)
        answers[key.strip()] = value.strip()
    return answers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default=SAMPLE_QUERY)
    parser.add_argument("--answer", action="append", default=[],
                        metavar="attr=value", help="supply an answer up front")
    parser.add_argument("--non-interactive", action="store_true",
                        help="never prompt; report gaps as unresolved instead")
    parser.add_argument("--override", action="store_true",
                        help="continue past an overridable stop, e.g. low fibre content")
    parser.add_argument("--anthropic", action="store_true")
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="hide per-node working")
    args = parser.parse_args()

    interactive = not args.non_interactive and sys.stdin.isatty()

    heading(f"PRODUCT: {args.query}")
    recorder = Recorder(verbose=not args.quiet)
    recorder.log({"type": "run_start", "query": args.query,
                  "interactive": interactive, "anthropic": args.anthropic})

    engine = build_engine(args, recorder)
    tree = engine.tree

    known = parse_answers(args.answer)
    unknown_keys = set(known) - set(tree.attributes)
    if unknown_keys:
        raise SystemExit(f"unknown attribute(s) in --answer: {sorted(unknown_keys)}")

    heading("TRAVERSAL")
    try:
        session = engine.start(args.query, known=known, interactive=interactive)
        if interactive:
            ask_user(session)
        if args.override and session.result and session.result.can_override:
            session.override()
            if interactive:
                ask_user(session)
    except ExtractionError as exc:
        print(f"\n  extraction failed: {exc}")
        recorder.log({"type": "error", "error": str(exc)})
        print(f"\n  Log saved → {recorder.save().relative_to(ROOT)}")
        return 1

    result = session.result
    assert result is not None
    print_result(result, tree)

    recorder.log({
        "type": "result",
        "method": result.method,
        "path": result.path,
        "confidence": result.confidence,
        "warning": result.warning,
        "unresolved": result.unresolved,
        "assumptions": result.assumptions,
        "attrs": {k: {"value": v.value, "confidence": v.confidence, "origin": v.origin}
                  for k, v in result.attrs.items()},
    })
    print(f"\n  Log saved → {recorder.save().relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
