"""
Test script: run the AOAC decision tree using Gemma 4 for attribute extraction
and the local RAG index for product context retrieval.

Usage:
    cd scripts/
    python3 test_gemma.py
    python3 test_gemma.py "whole wheat bread enriched with inulin"
"""

from __future__ import annotations

import json
import math
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from google import genai
from google.genai import types
import openai

from decision_tree_engine import DecisionTree, DecisionTreeEngine, TraversalResult, TreeNode

# ── Config ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
ENV = dotenv_values(ROOT / ".env.local")

GOOGLE_API_KEY = ENV.get("GOOGLE_GENERATIVE_AI_API_KEY", "")
OPENAI_API_KEY = ENV.get("OPENAI_API_KEY", "")
GEMMA_MODEL = "gemma-4-26b-a4b-it"
EMBED_MODEL = "text-embedding-3-small"  # must match the index

# ── Logging ───────────────────────────────────────────────────────────────────

_log_entries: list[dict] = []

def _log(entry: dict) -> None:
    entry["ts"] = datetime.now().isoformat(timespec="seconds")
    _log_entries.append(entry)

def _print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)

def _print_llm_call(kind: str, attributes: list[str], prompt_snippet: str, result: dict) -> None:
    _print_section(f"LLM CALL ({kind}) — {', '.join(attributes)}")
    print(f"Prompt (first 400 chars):\n{textwrap.indent(prompt_snippet[:400], '  ')}")
    print(f"\nExtracted:\n{textwrap.indent(json.dumps(result, indent=2), '  ')}")

def save_log(path: Path | None = None) -> Path:
    out = path or (ROOT / "scripts" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    with open(out, "w") as f:
        for entry in _log_entries:
            f.write(json.dumps(entry) + "\n")
    return out

# ── RAG retrieval (pure Python, reads pre-built index-parts) ─────────────────

_index_cache: list[dict] | None = None

def _load_index() -> list[dict]:
    global _index_cache
    if _index_cache is not None:
        return _index_cache
    chunks: list[dict] = []
    parts_dir = ROOT / "index-parts"
    for part_file in sorted(parts_dir.glob("index-part-*.json")):
        part = json.loads(part_file.read_text())
        chunks.extend(part["chunks"])
    _index_cache = chunks
    return chunks

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-12)

def rag_search(query: str, top_k: int = 5, max_chars_per_chunk: int = 400) -> str:
    """Embed query with OpenAI (same model as index), return context string.

    Chunks are truncated to max_chars_per_chunk to keep prompts within model limits.
    """
    _print_section(f"RAG RETRIEVAL — query: {query[:80]}")
    oa = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp = oa.embeddings.create(model=EMBED_MODEL, input=query)
    q_emb = resp.data[0].embedding

    chunks = _load_index()
    scored = sorted(chunks, key=lambda c: _cosine(q_emb, c["embedding"]), reverse=True)
    top = scored[:top_k]

    context_parts = []
    for i, chunk in enumerate(top, 1):
        section = chunk.get("section") or "General"
        score = _cosine(q_emb, chunk["embedding"])
        text = chunk["text"][:max_chars_per_chunk]
        print(f"  [{i}] {chunk['file']} — {section}  (score={score:.3f})")
        context_parts.append(f"[Source {i}][{section}]\n{text}")

    _log({"type": "rag", "query": query, "top_k": top_k,
          "results": [{"file": c["file"], "section": c.get("section"), "score": round(_cosine(q_emb, c["embedding"]), 4)} for c in top]})

    return "\n\n".join(context_parts)

# ── Gemma extractor ───────────────────────────────────────────────────────────

class GemmaDecisionTreeEngine(DecisionTreeEngine):
    """
    Drops-in over DecisionTreeEngine, swapping Claude tool-use for Gemma 4
    JSON-mode generation. Adds per-call logging.
    """

    def __init__(self, tree: DecisionTree, google_api_key: str, **kwargs) -> None:
        # Pass a dummy anthropic client — we never call it.
        import anthropic
        super().__init__(tree, anthropic.Anthropic(api_key="unused"), **kwargs)
        self._gemma = genai.Client(api_key=google_api_key)

    # ── Override extraction methods only ──────────────────────────────────────

    def _extract_one(self, node: TreeNode, context: str) -> Any:
        prompt = self._build_prompt([node], context)
        schema = {"properties": {node.attribute: self._gemma_schema(node)}}
        result = self._call_gemma(prompt, schema)
        value = result.get(node.attribute)
        _print_llm_call("single", [node.attribute], prompt, {node.attribute: value})
        _log({"type": "extract_one", "node": node.id, "attribute": node.attribute,
              "value": value, "prompt_snippet": prompt[:400]})
        print(f"\n  → {node.attribute} = {value}")
        return value

    def _extract_batch(self, nodes: list[TreeNode], context: str) -> dict[str, Any]:
        # Deduplicate by attribute key
        seen: set[str] = set()
        unique: list[TreeNode] = []
        for n in nodes:
            if n.attribute not in seen:
                seen.add(n.attribute)
                unique.append(n)

        prompt = self._build_prompt(unique, context)
        properties = {n.attribute: self._gemma_schema(n) for n in unique}
        schema = {"properties": properties}
        result = self._call_gemma(prompt, schema)
        _print_llm_call("batch", list(result.keys()), prompt, result)
        _log({"type": "extract_batch", "nodes": [n.id for n in unique],
              "attributes": list(result.keys()), "values": result,
              "prompt_snippet": prompt[:400]})
        for attr, val in result.items():
            print(f"  → {attr} = {val}")
        return result

    # ── Gemma call ────────────────────────────────────────────────────────────

    def _call_gemma(self, prompt: str, schema: dict, max_retries: int = 4) -> dict:
        # Gemma doesn't support response_schema — append the JSON spec to the
        # prompt and parse the output manually.
        field_specs = "\n".join(
            f'  "{k}": {self._schema_hint(v)}'
            for k, v in schema.get("properties", {}).items()
        )
        full_prompt = (
            f"{prompt}\n\n"
            f"Respond with ONLY a JSON object (no markdown, no explanation):\n"
            f"{{\n{field_specs}\n}}"
        )
        import time
        from google.genai import errors as genai_errors

        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self._gemma.models.generate_content(
                    model=GEMMA_MODEL,
                    contents=full_prompt,
                )
                raw = (response.text or "{}").strip()
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    import re
                    m = re.search(r'\{.*?\}', raw, re.DOTALL)
                    if m:
                        try:
                            return json.loads(m.group())
                        except json.JSONDecodeError:
                            pass
                    return {}
            except genai_errors.ServerError as exc:
                last_exc = exc
                wait = 2 ** attempt
                print(f"  [retry {attempt + 1}/{max_retries}] server error — waiting {wait}s …")
                time.sleep(wait)

        raise RuntimeError(f"Gemma API failed after {max_retries} retries") from last_exc

    def _schema_hint(self, schema: dict) -> str:
        if "enum" in schema:
            return f"one of {schema['enum']}"
        if schema.get("type") == "NUMBER":
            return "<number or null>"
        return '"<value>"'

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_prompt(self, nodes: list[TreeNode], context: str) -> str:
        questions = "\n".join(f"- {n.extraction_prompt.strip()}" for n in nodes)
        return (
            "You are a food science expert. Based ONLY on the product information below, "
            "answer each question as accurately as possible.\n\n"
            f"PRODUCT INFORMATION:\n{context}\n\n"
            f"QUESTIONS TO ANSWER:\n{questions}\n\n"
            "Respond in JSON using the exact field names specified."
        )

    def _gemma_schema(self, node: TreeNode) -> dict:
        if node.attribute_type == "boolean":
            return {"type": "STRING", "enum": ["yes", "no", "unknown"],
                    "description": node.question}
        if node.attribute_type == "numeric":
            return {"type": "NUMBER", "description": f"{node.question} (number or null if unknown)"}
        # categorical — derive from edge keys
        return {"type": "STRING", "enum": list(node.edges.keys()),
                "description": node.question}

# ── Main ──────────────────────────────────────────────────────────────────────

SAMPLE_QUERIES = [
    # "whole wheat bread enriched with inulin (FOS), sold ready-to-eat, EU market, lab needs separate IDF/SDF values",
    "canned kidney beans, requires heating before eating, no added NDO, high resistant starch content",
    # "fresh apple juice with no fibre enrichment, expected fibre 0.2 g/100g",
]

def run(product_query: str) -> TraversalResult:
    print(f"\n{'═' * 60}")
    print(f"  PRODUCT QUERY: {product_query}")
    print('═' * 60)

    _log({"type": "run_start", "query": product_query, "model": GEMMA_MODEL})

    # Step 1: RAG retrieval — scientific background from the corpus
    rag_context = rag_search(product_query)

    # Combine the user's product description (primary source of product facts)
    # with the RAG-retrieved scientific background.
    context = (
        f"PRODUCT DESCRIPTION (primary source for product-specific questions):\n"
        f"{product_query}\n\n"
        f"SUPPORTING SCIENTIFIC CONTEXT (from research corpus):\n"
        f"{rag_context}"
    )

    # Step 2: Decision tree traversal with Gemma
    tree = DecisionTree.from_yaml(Path(__file__).parent / "decision_tree.yaml")
    # batch_depth_threshold=99 → always lazy (one attribute per LLM call).
    # Gemma's context window is limited and the RAG context is already large.
    engine = GemmaDecisionTreeEngine(tree, GOOGLE_API_KEY, batch_depth_threshold=99)

    _print_section("DECISION TREE TRAVERSAL")
    result = engine.run(context)

    # Step 3: Print result
    _print_section("RESULT")
    print(f"  Recommended method : {result.method}")
    print(f"  Traversal path     : {' → '.join(result.path)}")
    if result.warning:
        print(f"  ⚠ Warning          : low fibre content — result may be unreliable")
    print(f"\n  All extracted attributes:")
    for k, v in result.attrs.items():
        print(f"    {k}: {v}")

    _log({"type": "result", "method": result.method, "path": result.path,
          "attrs": result.attrs, "warning": result.warning})

    return result


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else SAMPLE_QUERIES[0]
    result = run(query)

    log_path = save_log()
    print(f"\n  Log saved → {log_path.relative_to(ROOT)}")
