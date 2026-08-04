"""
Decision tree engine for AOAC method selection.

The engine walks a tree of questions about a product and returns the AOAC
method to use. Three properties matter more than the traversal itself:

1. It never invents a branch. Every node declares an `unknown` edge, and a
   value that does not match an edge raises instead of falling through to
   whichever branch happened to be listed first.
2. It asks rather than guesses. Attributes are tagged with who can answer them;
   anything the product description cannot settle — a labelling requirement, a
   jurisdiction's fibre definition — is put to the user instead of hallucinated.
3. It shows its work. Every answer carries a confidence and the literature it
   came from, and the result lists the assumptions it had to make.

    tree = DecisionTree.from_yaml("decision_tree.yaml")
    engine = DecisionTreeEngine(tree, extractor, retriever=retriever)
    session = engine.start("canned kidney beans, heat before eating")

    while session.pending:
        session.answer(input(session.pending.prompt + " "))

    print(session.result.method, session.result.unresolved)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import yaml

from extractors import Evidence, Extraction, ExtractionRequest, Extractor
from rag_index import Hit
from values import UNKNOWN, coerce_boolean, coerce_categorical, coerce_number

ATTRIBUTE_TYPES = {"boolean", "numeric", "categorical"}
ATTRIBUTE_SOURCES = {"product", "corpus", "user"}
DEFAULT_MIN_CONFIDENCE = 0.55


class TreeError(ValueError):
    """The tree definition is invalid, or traversal hit an impossible state."""


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AttributeSpec:
    key: str
    question: str
    type: str
    source: str
    ask: str
    why: str
    extraction_prompt: str = ""
    retrieval_query: str = ""
    options: list[str] = field(default_factory=list)

    @property
    def askable_only(self) -> bool:
        return self.source == "user"


@dataclass(frozen=True)
class Edge:
    condition: str
    type: str                          # "node" | "outcome"
    next: str | None = None
    result: str | None = None
    warning: bool = False
    warning_message: str = ""
    can_override: bool = False
    override_next: str | None = None
    assumption: str = ""


@dataclass(frozen=True)
class TreeNode:
    id: str
    question: str
    edges: dict[str, Edge]
    attribute: str | None = None
    all_of: list[str] = field(default_factory=list)
    threshold: float | None = None

    @property
    def attributes(self) -> list[str]:
        return self.all_of if self.all_of else ([self.attribute] if self.attribute else [])


@dataclass
class Question:
    """A question the engine is putting to the user rather than guessing at."""
    attribute: str
    prompt: str
    why: str
    type: str
    options: list[str] = field(default_factory=list)
    because_low_confidence: Extraction | None = None


@dataclass
class Step:
    """One node visited, and why it went the way it did."""
    node_id: str
    question: str
    value: Any
    extraction: Extraction | None
    edge: Edge


@dataclass
class TraversalResult:
    method: str
    path: list[str]
    steps: list[Step]
    attrs: dict[str, Extraction]
    warning: bool = False
    warning_message: str = ""
    can_override: bool = False
    override_next: str | None = None
    assumptions: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    confidence: float = 1.0

    @property
    def is_reliable(self) -> bool:
        return not self.unresolved and not self.warning

    def evidence(self) -> list[Evidence]:
        seen: set[tuple[str, str]] = set()
        out: list[Evidence] = []
        for step in self.steps:
            for ev in (step.extraction.evidence if step.extraction else []):
                key = (ev.file, ev.section)
                if key not in seen:
                    seen.add(key)
                    out.append(ev)
        return out


# ── Tree loader ───────────────────────────────────────────────────────────────

class DecisionTree:
    def __init__(self, data: dict) -> None:
        self.id: str = data.get("id", "unnamed")
        self.version: str = str(data.get("version", "0"))
        self.start: str = data.get("start", "")
        self.min_confidence: float = float(
            data.get("min_confidence", DEFAULT_MIN_CONFIDENCE)
        )

        self.attributes: dict[str, AttributeSpec] = {}
        for key, spec in (data.get("attributes") or {}).items():
            self.attributes[key] = AttributeSpec(
                key=key,
                question=str(spec.get("question", key)).strip(),
                type=str(spec.get("type", "boolean")).strip(),
                source=str(spec.get("source", "product")).strip(),
                ask=str(spec.get("ask", spec.get("question", key))).strip(),
                why=str(spec.get("why", "")).strip(),
                extraction_prompt=str(spec.get("extraction_prompt", "")).strip(),
                retrieval_query=str(spec.get("retrieval_query", "")).strip(),
                options=list(spec.get("options") or []),
            )

        self.nodes: dict[str, TreeNode] = {}
        for node_id, nd in (data.get("nodes") or {}).items():
            edges = {
                condition: Edge(
                    condition=condition,
                    type=ed.get("type", ""),
                    next=ed.get("next"),
                    result=(ed.get("result") or "").strip() or None,
                    warning=bool(ed.get("warning", False)),
                    warning_message=str(ed.get("warning_message", "")).strip(),
                    can_override=bool(ed.get("can_override", False)),
                    override_next=ed.get("override_next"),
                    assumption=str(ed.get("assumption", "")).strip(),
                )
                for condition, ed in (nd.get("edges") or {}).items()
            }
            attribute = nd.get("attribute")
            all_of = list(nd.get("all_of") or [])
            fallback_question = ""
            if attribute and attribute in self.attributes:
                fallback_question = self.attributes[attribute].question
            self.nodes[node_id] = TreeNode(
                id=node_id,
                question=str(nd.get("question") or fallback_question or node_id).strip(),
                edges=edges,
                attribute=attribute,
                all_of=all_of,
                threshold=nd.get("threshold"),
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> DecisionTree:
        with open(path) as f:
            return cls(yaml.safe_load(f))

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        errors: list[str] = []
        errors += self._validate_attributes()
        errors += self._validate_nodes()
        errors += self._validate_graph()
        return errors

    def _validate_attributes(self) -> list[str]:
        errors: list[str] = []
        if not 0.0 <= self.min_confidence <= 1.0:
            errors.append(f"min_confidence {self.min_confidence} is outside 0..1")
        for key, spec in self.attributes.items():
            if spec.type not in ATTRIBUTE_TYPES:
                errors.append(f"attribute '{key}': unknown type '{spec.type}'")
            if spec.source not in ATTRIBUTE_SOURCES:
                errors.append(f"attribute '{key}': unknown source '{spec.source}'")
            if spec.type == "categorical" and not spec.options:
                errors.append(f"attribute '{key}': categorical needs 'options'")
            if not spec.askable_only and not spec.extraction_prompt:
                errors.append(
                    f"attribute '{key}': source '{spec.source}' needs an extraction_prompt"
                )
            if not spec.ask:
                errors.append(f"attribute '{key}': needs an 'ask' phrasing")
            if not spec.why:
                errors.append(f"attribute '{key}': needs a 'why' explaining what it changes")
        return errors

    def _validate_nodes(self) -> list[str]:
        errors: list[str] = []
        for node_id, node in self.nodes.items():
            if bool(node.attribute) == bool(node.all_of):
                errors.append(f"{node_id}: needs exactly one of 'attribute' or 'all_of'")
                continue

            missing = [k for k in node.attributes if k not in self.attributes]
            if missing:
                errors.append(f"{node_id}: undefined attribute(s) {missing}")
                continue

            expected = self._expected_conditions(node, errors)
            actual = set(node.edges)
            if expected is not None:
                if UNKNOWN not in actual:
                    errors.append(
                        f"{node_id}: missing an '{UNKNOWN}' edge — without it an "
                        f"unanswered question silently takes an arbitrary branch"
                    )
                if actual != expected:
                    errors.append(
                        f"{node_id}: edges {sorted(actual)} do not match the "
                        f"expected {sorted(expected)}"
                    )

            errors += self._validate_edges(node_id, node)
        return errors

    def _expected_conditions(self, node: TreeNode, errors: list[str]) -> set[str] | None:
        if node.all_of:
            non_boolean = [
                k for k in node.all_of if self.attributes[k].type != "boolean"
            ]
            if non_boolean:
                errors.append(f"{node.id}: all_of members must be boolean, got {non_boolean}")
                return None
            return {"yes", "no", UNKNOWN}

        spec = self.attributes[node.attribute]  # type: ignore[index]
        if spec.type == "boolean":
            return {"yes", "no", UNKNOWN}
        if spec.type == "numeric":
            if node.threshold is None:
                errors.append(f"{node.id}: numeric attribute needs a 'threshold'")
                return None
            return {"above", "below", UNKNOWN}
        return {*spec.options, UNKNOWN}

    def _validate_edges(self, node_id: str, node: TreeNode) -> list[str]:
        errors: list[str] = []
        for condition, edge in node.edges.items():
            where = f"{node_id}.{condition}"
            if edge.type == "node":
                if not edge.next:
                    errors.append(f"{where}: 'next' is required for a node edge")
                elif edge.next not in self.nodes:
                    errors.append(f"{where}: next '{edge.next}' is not a node")
            elif edge.type == "outcome":
                if not edge.result:
                    errors.append(f"{where}: 'result' is required for an outcome edge")
                if edge.can_override:
                    if not edge.override_next:
                        errors.append(f"{where}: can_override needs an 'override_next'")
                    elif edge.override_next not in self.nodes:
                        errors.append(
                            f"{where}: override_next '{edge.override_next}' is not a node"
                        )
                if edge.warning and not edge.warning_message:
                    errors.append(f"{where}: a warning edge needs a 'warning_message'")
            else:
                errors.append(f"{where}: type must be 'node' or 'outcome', got '{edge.type}'")
        return errors

    def _validate_graph(self) -> list[str]:
        errors: list[str] = []
        if self.start not in self.nodes:
            errors.append(f"start node '{self.start}' not found")
            return errors

        cycle = self._find_cycle()
        if cycle:
            errors.append("cycle in tree: " + " → ".join(cycle))

        unreachable = sorted(set(self.nodes) - self._reachable())
        if unreachable:
            errors.append(f"unreachable node(s): {unreachable}")
        return errors

    def _successors(self, node_id: str) -> list[str]:
        out: list[str] = []
        for edge in self.nodes[node_id].edges.values():
            for target in (edge.next, edge.override_next):
                if target and target in self.nodes:
                    out.append(target)
        return out

    def _reachable(self) -> set[str]:
        seen: set[str] = set()
        queue = deque([self.start])
        while queue:
            node_id = queue.popleft()
            if node_id in seen or node_id not in self.nodes:
                continue
            seen.add(node_id)
            queue.extend(self._successors(node_id))
        return seen

    def _find_cycle(self) -> list[str] | None:
        colour: dict[str, int] = {}          # 0 = in progress, 1 = done

        def visit(node_id: str, stack: list[str]) -> list[str] | None:
            state = colour.get(node_id)
            if state == 0:
                return stack[stack.index(node_id):] + [node_id]
            if state == 1:
                return None
            colour[node_id] = 0
            for nxt in self._successors(node_id):
                found = visit(nxt, stack + [nxt])
                if found:
                    return found
            colour[node_id] = 1
            return None

        for node_id in self.nodes:
            found = visit(node_id, [node_id])
            if found:
                return found
        return None


# ── Retrieval ─────────────────────────────────────────────────────────────────

class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> Sequence[Hit]: ...


Event = Callable[[str, dict], None]


# ── Engine ────────────────────────────────────────────────────────────────────

class DecisionTreeEngine:
    def __init__(
        self,
        tree: DecisionTree,
        extractor: Extractor,
        retriever: Retriever | None = None,
        top_k: int = 6,
        on_event: Event | None = None,
    ) -> None:
        errors = tree.validate()
        if errors:
            raise TreeError("invalid decision tree:\n  " + "\n  ".join(errors))
        self.tree = tree
        self.extractor = extractor
        self.retriever = retriever
        self.top_k = top_k
        self.on_event = on_event or (lambda kind, payload: None)

    def start(
        self,
        product_description: str,
        known: dict[str, Any] | None = None,
        interactive: bool = True,
    ) -> Session:
        session = Session(self, product_description, interactive=interactive)
        for key, value in (known or {}).items():
            session.supply(key, value)
        session.advance()
        return session

    def run(
        self,
        product_description: str,
        known: dict[str, Any] | None = None,
    ) -> TraversalResult:
        """Non-interactive convenience wrapper.

        Anything that would have been asked is recorded as unresolved and shows
        up in `result.unresolved`, so the caller can still tell the difference
        between a supported answer and a defaulted one.
        """
        session = self.start(product_description, known=known, interactive=False)
        if session.result is None:  # pragma: no cover - non-interactive always finishes
            raise TreeError("non-interactive traversal did not produce a result")
        return session.result

    # ── Extraction ────────────────────────────────────────────────────────────

    def extract(self, spec: AttributeSpec, product_description: str) -> Extraction:
        hits: Sequence[Hit] = ()
        context = ""
        if self.retriever is not None:
            query = self._retrieval_query(spec, product_description)
            hits = self.retriever.search(query, self.top_k)
            from rag_index import build_context

            context = build_context(hits)
            self.on_event("retrieval", {"attribute": spec.key, "query": query, "hits": hits})

        request = ExtractionRequest(
            key=spec.key,
            question=spec.question,
            type=spec.type,
            instructions=spec.extraction_prompt or spec.question,
            product_description=product_description,
            options=spec.options or None,
            context=context,
            hits=hits,
        )
        extraction = self.extractor.extract(request)
        self.on_event("extraction", {"attribute": spec.key, "extraction": extraction})
        return extraction

    @staticmethod
    def _retrieval_query(spec: AttributeSpec, product_description: str) -> str:
        """Retrieve for the question being asked, not for the product alone.

        A single product-level retrieval reused across every node hands the
        model chunks about sample preparation when it is being asked about
        regulatory definitions. The product is still appended, trimmed, so the
        retrieval stays anchored to the food in question.
        """
        base = spec.retrieval_query or spec.question
        hint = " ".join(product_description.split())[:200]
        return f"{base.strip()} {hint}".strip()


# ── Session ───────────────────────────────────────────────────────────────────

class Session:
    """Drives one traversal, pausing whenever it needs a human answer.

    Answering re-walks the tree from the start. That is deliberate: extractions
    are cached per attribute, so a re-walk costs nothing, and it keeps the
    traversal a pure function of the answers gathered so far.
    """

    def __init__(
        self,
        engine: DecisionTreeEngine,
        product_description: str,
        interactive: bool = True,
    ) -> None:
        self.engine = engine
        self.tree = engine.tree
        self.product_description = product_description
        self.interactive = interactive
        self.attrs: dict[str, Extraction] = {}
        self.pending: Question | None = None
        self.result: TraversalResult | None = None
        self._asked: set[str] = set()
        self._overridden: set[str] = set()
        self._consulted: list[str] = []

    @property
    def done(self) -> bool:
        return self.result is not None

    # ── Driving ───────────────────────────────────────────────────────────────

    def advance(self) -> None:
        outcome = self._walk()
        if isinstance(outcome, Question):
            self.pending, self.result = outcome, None
        else:
            self.pending, self.result = None, outcome

    def answer(self, raw_value: Any) -> None:
        if self.pending is None:
            raise TreeError("no question is pending")
        self.supply(self.pending.attribute, raw_value)
        self.advance()

    def supply(self, attribute: str, raw_value: Any) -> None:
        """Record a human answer. Always wins over anything the model inferred."""
        spec = self.tree.attributes.get(attribute)
        if spec is None:
            raise TreeError(f"unknown attribute '{attribute}'")
        value = self._coerce(spec, raw_value)
        known = value is not None and value != UNKNOWN
        self.attrs[attribute] = Extraction(
            value=value,
            confidence=1.0 if known else 0.0,
            reasoning="supplied by the user",
            origin="user",
        )
        self._asked.add(attribute)
        self.engine.on_event("answer", {"attribute": attribute, "value": value})

    def override(self) -> None:
        """Continue past an overridable outcome (e.g. the low-fibre stop)."""
        if self.result is None or not self.result.can_override:
            raise TreeError("the current result cannot be overridden")
        self._overridden.add(self.result.path[-1])
        self.advance()

    # ── Traversal ─────────────────────────────────────────────────────────────

    def _walk(self) -> TraversalResult | Question:
        node_id = self.tree.start
        path: list[str] = []
        steps: list[Step] = []
        assumptions: list[str] = []
        seen: set[str] = set()
        self._consulted = []

        while True:
            if node_id in seen:
                raise TreeError(f"traversal revisited '{node_id}' — the tree has a cycle")
            seen.add(node_id)
            path.append(node_id)
            node = self.tree.nodes[node_id]

            resolved = self._resolve_node(node)
            if isinstance(resolved, Question):
                return resolved
            condition, extraction = resolved

            edge = node.edges.get(condition)
            if edge is None:
                # Validation guarantees an edge for every condition the
                # resolvers can produce; reaching here means the tree and the
                # engine disagree, which must never be papered over.
                raise TreeError(
                    f"{node_id}: no edge for '{condition}' "
                    f"(have {sorted(node.edges)})"
                )

            steps.append(Step(
                node_id=node_id,
                question=node.question,
                value=extraction.value if extraction else condition,
                extraction=extraction,
                edge=edge,
            ))
            if edge.assumption:
                assumptions.append(edge.assumption)

            if edge.type == "outcome":
                if edge.can_override and node_id in self._overridden and edge.override_next:
                    assumptions.append(
                        f"Continued past the stop at '{node_id}' at the user's request."
                    )
                    node_id = edge.override_next
                    continue
                return self._finish(edge, path, steps, assumptions)

            node_id = edge.next  # type: ignore[assignment]

    def _finish(
        self,
        edge: Edge,
        path: list[str],
        steps: list[Step],
        assumptions: list[str],
    ) -> TraversalResult:
        used = [s.extraction for s in steps if s.extraction is not None]
        # Only attributes the walk actually consulted count as unresolved. A
        # conjunction that short-circuited never needed its later halves, and
        # reporting those as gaps would send the user chasing answers that
        # cannot change the recommendation.
        unresolved = [
            key
            for key in dict.fromkeys(self._consulted)
            if not self.attrs.get(key, Extraction(None)).trusted(self.tree.min_confidence)
        ]
        confidence = min((e.confidence for e in used), default=1.0)

        return TraversalResult(
            method=edge.result or "",
            path=path,
            steps=steps,
            attrs=dict(self.attrs),
            warning=edge.warning,
            warning_message=edge.warning_message,
            can_override=edge.can_override,
            override_next=edge.override_next,
            assumptions=assumptions,
            unresolved=unresolved,
            confidence=confidence,
        )

    # ── Node resolution ───────────────────────────────────────────────────────

    def _resolve_node(self, node: TreeNode) -> tuple[str, Extraction | None] | Question:
        if node.all_of:
            return self._resolve_all_of(node)

        resolved = self._ensure(node.attribute)  # type: ignore[arg-type]
        if isinstance(resolved, Question):
            return resolved

        extraction = resolved
        spec = self.tree.attributes[node.attribute]  # type: ignore[index]
        if not extraction.trusted(self.tree.min_confidence):
            return UNKNOWN, extraction
        if spec.type == "numeric":
            threshold = node.threshold or 0.0
            return ("above" if float(extraction.value) > threshold else "below"), extraction
        return str(extraction.value), extraction

    def _resolve_all_of(self, node: TreeNode) -> tuple[str, Extraction] | Question:
        """Evaluate the conjunction left to right, short-circuiting on a `no`.

        Order in the YAML is load-bearing: cheap inferable parts come first, so
        a product that plainly contains no NDO never triggers the jurisdiction
        question, which the user would rightly find pointless.
        """
        parts: list[Extraction] = []
        for key in node.all_of:
            resolved = self._ensure(key)
            if isinstance(resolved, Question):
                return resolved
            parts.append(resolved)
            if resolved.trusted(self.tree.min_confidence) and resolved.value == "no":
                break
        return self._combine_all_of(parts)

    def _combine_all_of(self, parts: list[Extraction]) -> tuple[str, Extraction]:
        trusted = [p.trusted(self.tree.min_confidence) for p in parts]
        values = [p.value if ok else UNKNOWN for p, ok in zip(parts, trusted)]

        if values and all(v == "yes" for v in values):
            condition = "yes"
        elif any(v == "no" for v in values):
            condition = "no"
        else:
            condition = UNKNOWN

        merged = Extraction(
            value=condition,
            confidence=min((p.confidence for p in parts), default=0.0),
            reasoning="; ".join(f"{p.value}: {p.reasoning}" for p in parts if p.reasoning),
            evidence=[ev for p in parts for ev in p.evidence],
            origin="derived",
        )
        return condition, merged

    def _ensure(self, key: str) -> Extraction | Question:
        spec = self.tree.attributes[key]
        existing = self.attrs.get(key)
        self._consulted.append(key)

        if existing is not None and existing.trusted(self.tree.min_confidence):
            return existing
        if key in self._asked:
            # Already put to the user; their "don't know" stands.
            return existing or self._unknown(spec, "user")

        if spec.askable_only:
            if self.interactive:
                return self._question(spec)
            return self._record(key, self._unknown(spec, "default"))

        if existing is None:
            existing = self._record(key, self.engine.extract(spec, self.product_description))

        if existing.trusted(self.tree.min_confidence):
            return existing
        if self.interactive:
            return self._question(spec, because=existing)
        return existing

    def _record(self, key: str, extraction: Extraction) -> Extraction:
        self.attrs[key] = extraction
        return extraction

    def _question(self, spec: AttributeSpec, because: Extraction | None = None) -> Question:
        return Question(
            attribute=spec.key,
            prompt=spec.ask,
            why=spec.why,
            type=spec.type,
            options=self._options_for(spec),
            because_low_confidence=because,
        )

    @staticmethod
    def _options_for(spec: AttributeSpec) -> list[str]:
        if spec.type == "boolean":
            return ["yes", "no", UNKNOWN]
        if spec.type == "categorical":
            return [*spec.options, UNKNOWN]
        return []

    @staticmethod
    def _unknown(spec: AttributeSpec, origin: str) -> Extraction:
        value = None if spec.type == "numeric" else UNKNOWN
        return Extraction(value=value, confidence=0.0, reasoning="not established",
                          origin=origin)

    @staticmethod
    def _coerce(spec: AttributeSpec, raw_value: Any) -> Any:
        if spec.type == "boolean":
            return coerce_boolean(raw_value)
        if spec.type == "numeric":
            return coerce_number(raw_value)
        return coerce_categorical(raw_value, spec.options)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tree = DecisionTree.from_yaml(Path(__file__).parent / "decision_tree.yaml")
    problems = tree.validate()
    if problems:
        print("Tree validation errors:")
        for problem in problems:
            print(" ", problem)
        raise SystemExit(1)
    print(
        f"Tree '{tree.id}' v{tree.version} is valid — "
        f"{len(tree.nodes)} nodes, {len(tree.attributes)} attributes, "
        f"min_confidence {tree.min_confidence}"
    )
    for key, spec in tree.attributes.items():
        print(f"  {spec.source:8s} {key}")
