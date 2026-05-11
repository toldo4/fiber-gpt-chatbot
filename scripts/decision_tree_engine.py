"""
Decision tree engine for AOAC method selection.

Usage:
    import anthropic
    from decision_tree_engine import DecisionTree, DecisionTreeEngine

    tree = DecisionTree.from_yaml("decision_tree.yaml")
    engine = DecisionTreeEngine(tree, anthropic.Anthropic())
    result = engine.run(product_context="Whole wheat bread, not enriched...")
    print(result.method)   # e.g. "AOAC 991.43"
    print(result.path)     # nodes visited
    print(result.attrs)    # all extracted attributes
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
import yaml

# When the remaining subtree depth is at or above this value, batch-extract
# all attributes needed for the subtree in one LLM call instead of one-by-one.
DEFAULT_BATCH_DEPTH_THRESHOLD = 3


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Edge:
    type: str           # "node" | "outcome"
    next: str | None = None
    result: str | None = None
    warning: bool = False
    can_override: bool = False  # outcome can be bypassed; caller handles this


@dataclass
class TreeNode:
    id: str
    question: str
    attribute: str
    attribute_type: str     # "boolean" | "numeric" | "categorical"
    extraction_prompt: str
    edges: dict[str, Edge]
    threshold: float | None = None  # only for attribute_type == "numeric"


@dataclass
class TraversalResult:
    method: str
    path: list[str]
    attrs: dict[str, Any]
    warning: bool = False
    can_override: bool = False


# ── Tree loader ───────────────────────────────────────────────────────────────

class DecisionTree:
    def __init__(self, data: dict) -> None:
        self.id: str = data["id"]
        self.version: str = data["version"]
        self.start: str = data["start"]
        self.nodes: dict[str, TreeNode] = {}

        for node_id, nd in data["nodes"].items():
            edges = {
                cond: Edge(
                    type=ed["type"],
                    next=ed.get("next"),
                    result=ed.get("result"),
                    warning=ed.get("warning", False),
                    can_override=ed.get("can_override", False),
                )
                for cond, ed in nd["edges"].items()
            }
            self.nodes[node_id] = TreeNode(
                id=node_id,
                question=nd["question"],
                attribute=nd["attribute"],
                attribute_type=nd["attribute_type"],
                extraction_prompt=nd["extraction_prompt"],
                edges=edges,
                threshold=nd.get("threshold"),
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> DecisionTree:
        with open(path) as f:
            return cls(yaml.safe_load(f))

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.start not in self.nodes:
            errors.append(f"start node '{self.start}' not found")
        for node_id, node in self.nodes.items():
            for cond, edge in node.edges.items():
                if edge.type == "node":
                    if not edge.next:
                        errors.append(f"{node_id}.{cond}: 'next' required for node edge")
                    elif edge.next not in self.nodes:
                        errors.append(f"{node_id}.{cond}: next '{edge.next}' not found")
                elif edge.type == "outcome":
                    if not edge.result:
                        errors.append(f"{node_id}.{cond}: 'result' required for outcome edge")
        return errors


# ── Engine ────────────────────────────────────────────────────────────────────

class DecisionTreeEngine:
    """
    Traverses a DecisionTree, extracting product attributes via Claude when needed.

    Extraction strategy (middle-ground):
    - If the remaining subtree depth >= batch_depth_threshold → extract all
      attributes needed for the entire subtree in ONE LLM call (batch).
    - Otherwise → extract just the current node's attribute (lazy).

    Because attributes are cached in `attrs`, shared attribute keys (e.g.
    `needs_idf_sdf_separate`) are only extracted once even if they appear in
    multiple nodes.
    """

    def __init__(
        self,
        tree: DecisionTree,
        client: anthropic.Anthropic,
        model: str = "claude-sonnet-4-6",
        batch_depth_threshold: int = DEFAULT_BATCH_DEPTH_THRESHOLD,
    ) -> None:
        self.tree = tree
        self.client = client
        self.model = model
        self.batch_depth_threshold = batch_depth_threshold

    def run(self, product_context: str) -> TraversalResult:
        errors = self.tree.validate()
        if errors:
            raise ValueError("Invalid decision tree:\n" + "\n".join(errors))

        attrs: dict[str, Any] = {}
        return self._step(self.tree.start, attrs, product_context, [])

    # ── Traversal ─────────────────────────────────────────────────────────────

    def _step(
        self,
        node_id: str,
        attrs: dict[str, Any],
        context: str,
        path: list[str],
    ) -> TraversalResult:
        node = self.tree.nodes[node_id]
        current_path = path + [node_id]

        if node.attribute not in attrs:
            depth = self._subtree_depth(node_id)
            if depth >= self.batch_depth_threshold:
                needed = [
                    n for n in self._subtree_nodes(node_id)
                    if n.attribute not in attrs
                ]
                if needed:
                    attrs.update(self._extract_batch(needed, context))
            else:
                attrs[node.attribute] = self._extract_one(node, context)

        edge = self._resolve_edge(node, attrs.get(node.attribute))

        if edge.type == "outcome":
            return TraversalResult(
                method=edge.result or "",
                path=current_path,
                attrs=dict(attrs),
                warning=edge.warning,
                can_override=edge.can_override,
            )

        return self._step(edge.next, attrs, context, current_path)  # type: ignore[arg-type]

    def _resolve_edge(self, node: TreeNode, value: Any) -> Edge:
        if node.attribute_type == "numeric":
            if value is None:
                return node.edges.get("unknown", next(iter(node.edges.values())))
            key = "above" if float(value) > (node.threshold or 0) else "below"
            return node.edges.get(key, next(iter(node.edges.values())))

        str_val = str(value).lower() if value is not None else "unknown"
        return node.edges.get(str_val, node.edges.get("unknown", next(iter(node.edges.values()))))

    # ── Subtree helpers (pure computation, no LLM) ────────────────────────────

    def _subtree_depth(self, start: str) -> int:
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        max_depth = 0
        while queue:
            node_id, depth = queue.popleft()
            if node_id in visited or node_id not in self.tree.nodes:
                continue
            visited.add(node_id)
            max_depth = max(max_depth, depth)
            for edge in self.tree.nodes[node_id].edges.values():
                if edge.type == "node" and edge.next:
                    queue.append((edge.next, depth + 1))
        return max_depth

    def _subtree_nodes(self, start: str) -> list[TreeNode]:
        visited: set[str] = set()
        queue: deque[str] = deque([start])
        nodes: list[TreeNode] = []
        while queue:
            node_id = queue.popleft()
            if node_id in visited or node_id not in self.tree.nodes:
                continue
            visited.add(node_id)
            node = self.tree.nodes[node_id]
            nodes.append(node)
            for edge in node.edges.values():
                if edge.type == "node" and edge.next:
                    queue.append(edge.next)
        return nodes

    # ── LLM extraction ────────────────────────────────────────────────────────

    def _extract_one(self, node: TreeNode, context: str) -> Any:
        """Single-attribute extraction via tool use."""
        schema = self._attribute_schema(node)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            tools=[{
                "name": "extract_attribute",
                "description": f"Extract the answer to: {node.question}",
                "input_schema": {
                    "type": "object",
                    "properties": {node.attribute: schema},
                    "required": [node.attribute],
                },
            }],
            tool_choice={"type": "tool", "name": "extract_attribute"},
            messages=[{
                "role": "user",
                "content": (
                    f"{node.extraction_prompt}\n\n"
                    f"Product information:\n{context}"
                ),
            }],
        )
        for block in response.content:
            if block.type == "tool_use":
                return block.input.get(node.attribute)
        return None

    def _extract_batch(self, nodes: list[TreeNode], context: str) -> dict[str, Any]:
        """
        Extract multiple attributes in one LLM call.
        Deduplicates by attribute key so the same question is only included once
        even if multiple nodes share an attribute.
        """
        seen: set[str] = set()
        unique_nodes: list[TreeNode] = []
        for n in nodes:
            if n.attribute not in seen:
                seen.add(n.attribute)
                unique_nodes.append(n)

        properties = {n.attribute: self._attribute_schema(n) for n in unique_nodes}
        questions = "\n".join(f"- {n.extraction_prompt}" for n in unique_nodes)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            tools=[{
                "name": "extract_attributes",
                "description": "Extract multiple product attributes at once.",
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties.keys()),
                },
            }],
            tool_choice={"type": "tool", "name": "extract_attributes"},
            messages=[{
                "role": "user",
                "content": (
                    f"Answer each of the following questions about this product:\n{questions}\n\n"
                    f"Product information:\n{context}"
                ),
            }],
        )
        for block in response.content:
            if block.type == "tool_use":
                return block.input  # type: ignore[return-value]
        return {}

    def _attribute_schema(self, node: TreeNode) -> dict:
        if node.attribute_type == "boolean":
            return {
                "type": "string",
                "enum": ["yes", "no", "unknown"],
                "description": node.question,
            }
        if node.attribute_type == "numeric":
            return {
                "type": ["number", "null"],
                "description": f"{node.question} (numeric value or null if unknown)",
            }
        # categorical — derive enum from edge keys
        return {
            "type": "string",
            "enum": list(node.edges.keys()),
            "description": node.question,
        }


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    tree = DecisionTree.from_yaml(Path(__file__).parent / "decision_tree.yaml")
    errors = tree.validate()
    if errors:
        print("Tree validation errors:")
        for e in errors:
            print(" ", e)
    else:
        print(f"Tree '{tree.id}' v{tree.version} validated OK — {len(tree.nodes)} nodes")

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    engine = DecisionTreeEngine(tree, client)

    sample_context = (
        "Product: Cooked kidney beans (canned, ready-to-eat after heating). "
        "Expected dietary fibre content: ~6 g/100g. "
        "Contains resistant starch. No added NDO ingredients. "
        "Local regulation (EU) does not include NDO in the fibre definition. "
        "Lab requires separate IDF and SDF values for nutrition labelling."
    )

    result = engine.run(sample_context)
    print("\nResult:")
    print(f"  Method : {result.method}")
    print(f"  Path   : {' → '.join(result.path)}")
    print(f"  Warning: {result.warning}")
    print(f"  Attrs  : {json.dumps(result.attrs, indent=4)}")
