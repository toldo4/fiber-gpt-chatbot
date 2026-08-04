"""
Attribute extractors.

The traversal engine depends on the `Extractor` protocol only, so swapping
models is a constructor argument rather than a subclass that has to fake a
client it never calls.

Every extractor returns an `Extraction`, not a bare value: the engine needs the
confidence to decide whether to trust an answer or ask the user, and the
evidence to explain the recommendation afterwards.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from rag_index import Hit
from values import UNKNOWN, coerce_boolean, coerce_categorical, coerce_number


class ExtractionError(RuntimeError):
    """The model could not be reached, or returned something unparseable.

    Deliberately distinct from an answer of "unknown": a transport or parsing
    failure must not quietly become a confident branch decision.
    """


@dataclass(frozen=True)
class Evidence:
    file: str
    section: str
    quote: str

    def __str__(self) -> str:
        return f"{self.file} — {self.section}"


@dataclass
class Extraction:
    value: Any
    confidence: float = 0.0
    reasoning: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    origin: str = "model"        # model | user | derived | default

    @property
    def is_known(self) -> bool:
        return self.value is not None and self.value != UNKNOWN

    def trusted(self, min_confidence: float) -> bool:
        return self.is_known and self.confidence >= min_confidence


@dataclass
class ExtractionRequest:
    key: str
    question: str
    type: str                       # boolean | numeric | categorical
    instructions: str
    product_description: str
    options: list[str] | None = None
    context: str = ""
    hits: Sequence[Hit] = ()


class Extractor(Protocol):
    def extract(self, request: ExtractionRequest) -> Extraction: ...


# ── Shared prompt construction ────────────────────────────────────────────────

_ANSWER_RULES = """\
How to answer:
- Facts about this specific sample come from the PRODUCT DESCRIPTION. The
  literature is for general scientific facts about this class of food, or about
  the analytical methods themselves.
- If neither settles the question, answer "unknown". "unknown" is a useful
  answer that will cause a human to be asked; a confident wrong answer is not.
- "confidence" is how likely you think it is that your value is correct, from
  0 to 1. Use a low value when you are extrapolating.
- "sources" lists the [Source N] numbers you actually relied on, or [] if the
  answer came from the product description alone."""


def _value_spec(request: ExtractionRequest) -> str:
    if request.type == "boolean":
        return '"yes", "no", or "unknown"'
    if request.type == "numeric":
        return "a number, or null if it is not known"
    options = ", ".join(f'"{o}"' for o in (request.options or []))
    return f"one of {options}, or \"unknown\""


def build_task_prompt(request: ExtractionRequest) -> str:
    sections = [
        "You are a food-analysis expert selecting an AOAC dietary fibre method.",
        f"PRODUCT DESCRIPTION:\n{request.product_description.strip()}",
    ]
    if request.context.strip():
        sections.append(
            "RETRIEVED LITERATURE (curated dietary fibre method corpus):\n"
            f"{request.context.strip()}"
        )
    sections.append(f"QUESTION:\n{request.instructions.strip()}")
    sections.append(_ANSWER_RULES)
    return "\n\n".join(sections)


def json_format_block(request: ExtractionRequest) -> str:
    return (
        "Respond with ONLY a JSON object, no markdown fence and no commentary:\n"
        "{\n"
        f'  "value": {_value_spec(request)},\n'
        '  "confidence": <number between 0 and 1>,\n'
        '  "reasoning": "<one or two sentences>",\n'
        '  "sources": [<source numbers you used>]\n'
        "}"
    )


def parse_json_object(raw: str) -> dict:
    """Pull the JSON object out of a model response.

    Scans from the first brace to the last, not the first balanced-looking pair:
    a non-greedy match stops at the first inner closing brace and silently
    mangles any nested object.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ExtractionError(f"no JSON object in model response: {raw[:200]!r}")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"malformed JSON in model response: {raw[:200]!r}") from exc


def finalize(request: ExtractionRequest, payload: dict) -> Extraction:
    """Coerce a raw model payload into a validated Extraction."""
    raw_value = payload.get("value")
    if request.type == "boolean":
        value: Any = coerce_boolean(raw_value)
    elif request.type == "numeric":
        value = coerce_number(raw_value)
    else:
        value = coerce_categorical(raw_value, request.options or [])

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)

    known = value is not None and value != UNKNOWN
    if not known:
        confidence = 0.0

    return Extraction(
        value=value,
        confidence=confidence,
        reasoning=str(payload.get("reasoning", "")).strip(),
        evidence=_resolve_evidence(request.hits, payload.get("sources")),
        origin="model",
    )


def _resolve_evidence(hits: Sequence[Hit], cited: Any, quote_chars: int = 280) -> list[Evidence]:
    if not hits or not isinstance(cited, (list, tuple)):
        return []
    evidence: list[Evidence] = []
    for entry in cited:
        try:
            index = int(entry) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(hits):
            hit = hits[index]
            quote = " ".join(hit.text.split())[:quote_chars]
            evidence.append(Evidence(file=hit.file, section=hit.section, quote=quote))
    return evidence


# ── Gemma (Gemini API) ────────────────────────────────────────────────────────

# Gemma 4 is a thinking model and its reasoning tokens are drawn from the same
# max_output_tokens budget as the answer, but the budget itself cannot be split
# (the API rejects thinking_config for this model). On a full extraction prompt
# it spends 1500-2000 tokens thinking, so a budget sized for the ~90-token JSON
# answer produces finish_reason=MAX_TOKENS and an empty response.
GEMMA_OUTPUT_TOKENS = 4096
GEMMA_OUTPUT_TOKENS_CEILING = 16384


class GemmaExtractor:
    """Gemma has no structured-output mode, so the JSON shape goes in the prompt.

    A parse failure raises rather than returning an empty dict — an empty dict
    reads downstream as "unknown", which is indistinguishable from the model
    genuinely not knowing and hides a broken pipeline.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemma-4-26b-a4b-it",
        max_retries: int = 4,
        max_output_tokens: int = GEMMA_OUTPUT_TOKENS,
        on_retry=None,
    ) -> None:
        from google import genai

        self.model = model
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
        self._client = genai.Client(api_key=api_key)
        self._on_retry = on_retry

    def extract(self, request: ExtractionRequest) -> Extraction:
        prompt = f"{build_task_prompt(request)}\n\n{json_format_block(request)}"
        payload = self._complete(prompt)
        return finalize(request, payload)

    def _complete(self, prompt: str) -> dict:
        from google.genai import errors as genai_errors
        from google.genai import types

        budget = self.max_output_tokens
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            wait = 2**attempt
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0, max_output_tokens=budget
                    ),
                )
                text = (response.text or "").strip()
                if text:
                    return parse_json_object(text)

                reason = self._finish_reason(response)
                if reason == "MAX_TOKENS" and budget < GEMMA_OUTPUT_TOKENS_CEILING:
                    # Retrying with the same budget would fail identically, so
                    # grow it and go again immediately rather than backing off.
                    budget = min(budget * 2, GEMMA_OUTPUT_TOKENS_CEILING)
                    last_error = ExtractionError(
                        f"answer budget exhausted by reasoning tokens; "
                        f"retrying with max_output_tokens={budget}"
                    )
                    wait = 0
                else:
                    last_error = ExtractionError(
                        f"Gemma returned no text (finish_reason={reason})"
                    )
            except genai_errors.ServerError as exc:
                last_error = exc
            except genai_errors.ClientError as exc:
                # Rate limits are worth waiting out; a 400 never is.
                if getattr(exc, "code", None) != 429:
                    raise ExtractionError(f"Gemma rejected the request: {exc}") from exc
                last_error = exc
            except ExtractionError as exc:
                last_error = exc

            if attempt < self.max_retries - 1:
                if self._on_retry:
                    self._on_retry(attempt + 1, self.max_retries, wait, last_error)
                if wait:
                    time.sleep(wait)

        raise ExtractionError(
            f"Gemma failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _finish_reason(response) -> str:
        try:
            reason = response.candidates[0].finish_reason
        except (AttributeError, IndexError, TypeError):
            return "unknown"
        return getattr(reason, "name", str(reason))


# ── Anthropic ─────────────────────────────────────────────────────────────────

class AnthropicExtractor:
    """Same contract, using tool use to enforce the response shape."""

    def __init__(self, client, model: str = "claude-sonnet-5", max_tokens: int = 768) -> None:
        self._client = client
        self.model = model
        self.max_tokens = max_tokens

    def extract(self, request: ExtractionRequest) -> Extraction:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            tools=[{
                "name": "record_answer",
                "description": f"Record the answer to: {request.question}",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "value": self._value_schema(request),
                        "confidence": {
                            "type": "number",
                            "description": "Probability the value is correct, 0 to 1.",
                        },
                        "reasoning": {"type": "string"},
                        "sources": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "The [Source N] numbers actually relied on.",
                        },
                    },
                    "required": ["value", "confidence", "reasoning", "sources"],
                },
            }],
            tool_choice={"type": "tool", "name": "record_answer"},
            messages=[{"role": "user", "content": build_task_prompt(request)}],
        )
        for block in response.content:
            if block.type == "tool_use":
                return finalize(request, dict(block.input))
        raise ExtractionError("model returned no tool_use block")

    @staticmethod
    def _value_schema(request: ExtractionRequest) -> dict:
        if request.type == "boolean":
            return {"type": "string", "enum": ["yes", "no", "unknown"],
                    "description": request.question}
        if request.type == "numeric":
            return {"type": ["number", "null"],
                    "description": f"{request.question} (null if unknown)"}
        return {"type": "string", "enum": [*(request.options or []), UNKNOWN],
                "description": request.question}
