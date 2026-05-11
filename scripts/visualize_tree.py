"""
Visualize the AOAC decision tree.

Outputs:
  decision_tree.mmd  — Mermaid flowchart (paste into mermaid.live or VS Code Mermaid extension)
  decision_tree.png  — Matplotlib figure

Usage:
    python3 visualize_tree.py
    python3 visualize_tree.py --mermaid-only
"""

from __future__ import annotations

import sys
import textwrap
from collections import deque
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from decision_tree_engine import DecisionTree

TREE_YAML = Path(__file__).parent / "decision_tree.yaml"
OUT_DIR   = Path(__file__).parent

# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(tree: DecisionTree) -> tuple[nx.DiGraph, dict[str, str]]:
    """
    Returns (G, outcome_map) where outcome_map maps result text → synthetic node id.
    Shared results (same AOAC method from multiple paths) reuse the same outcome node
    so the diagram shows convergence.
    """
    G = nx.DiGraph()

    for node_id, node in tree.nodes.items():
        G.add_node(node_id, kind="question", label=node.question)

    outcome_counter = 0
    outcome_map: dict[str, str] = {}

    for node_id, node in tree.nodes.items():
        for cond, edge in node.edges.items():
            if edge.type == "node":
                G.add_edge(node_id, edge.next, label=cond)
            elif edge.type == "outcome":
                result = (edge.result or "").strip()
                if result not in outcome_map:
                    outcome_counter += 1
                    oid = f"_out{outcome_counter}"
                    outcome_map[result] = oid
                    G.add_node(oid, kind="warning" if edge.warning else "outcome", label=result)
                G.add_edge(node_id, outcome_map[result], label=cond)

    return G, outcome_map


def _bfs_layers(G: nx.DiGraph, root: str) -> dict[str, int]:
    layers: dict[str, int] = {root: 0}
    q: deque[str] = deque([root])
    while q:
        node = q.popleft()
        for child in G.successors(node):
            if child not in layers:
                layers[child] = layers[node] + 1
                q.append(child)
    return layers

# ── PNG via matplotlib ────────────────────────────────────────────────────────

COLORS = {
    "question": "#aec6f7",
    "outcome":  "#a8d8a8",
    "warning":  "#ffe08a",
}
STROKE = {
    "question": "#3a7bd5",
    "outcome":  "#28a745",
    "warning":  "#e6a817",
}

def _wrap(text: str, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(text.strip(), width=width))


def save_png(tree: DecisionTree, path: Path) -> None:
    G, _ = build_graph(tree)
    layers = _bfs_layers(G, tree.start)
    for n, l in layers.items():
        G.nodes[n]["layer"] = l

    pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal", scale=2.5)
    # multipartite_layout puts layer 0 at bottom — flip y so root is at top
    pos = {n: (x, -y) for n, (x, y) in pos.items()}

    fig, ax = plt.subplots(figsize=(24, 16))
    fig.patch.set_facecolor("#f5f7fa")
    ax.set_facecolor("#f5f7fa")

    for kind in ("question", "outcome", "warning"):
        nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == kind]
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes, ax=ax,
            node_color=COLORS[kind],
            node_size=4500,
            node_shape="o",
            linewidths=1.5,
            edgecolors=STROKE[kind],
        )

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        arrows=True,
        arrowsize=18,
        arrowstyle="-|>",
        edge_color="#666666",
        width=1.2,
        connectionstyle="arc3,rad=0.05",
        min_source_margin=30,
        min_target_margin=30,
    )

    labels = {n: _wrap(d.get("label", n)) for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6.5, ax=ax, font_weight="normal")

    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels,
        font_size=6, ax=ax, font_color="#333333",
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "alpha": 0.7, "ec": "none"},
    )

    legend_handles = [
        mpatches.Patch(facecolor=COLORS["question"], edgecolor=STROKE["question"], label="Decision node"),
        mpatches.Patch(facecolor=COLORS["outcome"],  edgecolor=STROKE["outcome"],  label="AOAC method outcome"),
        mpatches.Patch(facecolor=COLORS["warning"],  edgecolor=STROKE["warning"],  label="Warning / not recommended"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title(f"AOAC Method Decision Tree — {tree.id}  v{tree.version}",
                 fontsize=14, pad=16, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"PNG  → {path.relative_to(OUT_DIR.parent)}")
    plt.show()

# ── Mermaid ───────────────────────────────────────────────────────────────────

def to_mermaid(tree: DecisionTree) -> str:
    def esc(text: str, n: int = 48) -> str:
        text = text.strip().replace('"', "'")
        return text if len(text) <= n else text[:n - 3] + "..."

    lines = ["flowchart TD"]
    outcome_counter = 0
    outcome_nodes: dict[str, tuple[str, bool]] = {}  # result → (oid, is_warning)

    # Question nodes
    for node_id, node in tree.nodes.items():
        lines.append(f'    {node_id}["{esc(node.question)}"]')

    lines.append("")

    # Edges + synthetic outcome nodes
    for node_id, node in tree.nodes.items():
        for cond, edge in node.edges.items():
            if edge.type == "node":
                lines.append(f'    {node_id} -->|{cond}| {edge.next}')
            elif edge.type == "outcome":
                result = (edge.result or "").strip()
                if result not in outcome_nodes:
                    outcome_counter += 1
                    oid = f"out{outcome_counter}"
                    outcome_nodes[result] = (oid, edge.warning)
                    label = esc(result)
                    shape = f'{{"{label}"}}' if edge.warning else f'(["{label}"])'
                    lines.append(f'    {oid}{shape}')
                oid = outcome_nodes[result][0]
                lines.append(f'    {node_id} -->|{cond}| {oid}')

    lines.append("")

    # Styles
    lines.append("    classDef question fill:#aec6f7,stroke:#3a7bd5,color:#000,rx:4")
    lines.append("    classDef outcome  fill:#a8d8a8,stroke:#28a745,color:#000")
    lines.append("    classDef warning  fill:#ffe08a,stroke:#e6a817,color:#000")

    q_ids   = " ".join(tree.nodes.keys())
    out_ids = " ".join(oid for oid, warn in outcome_nodes.values() if not warn)
    wrn_ids = " ".join(oid for oid, warn in outcome_nodes.values() if warn)
    lines.append(f"    class {q_ids} question")
    if out_ids:
        lines.append(f"    class {out_ids} outcome")
    if wrn_ids:
        lines.append(f"    class {wrn_ids} warning")

    return "\n".join(lines)

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mermaid_only = "--mermaid-only" in sys.argv

    tree = DecisionTree.from_yaml(TREE_YAML)
    errors = tree.validate()
    if errors:
        print("Tree validation errors:")
        for e in errors:
            print(" ", e)
        sys.exit(1)

    # Mermaid
    mmd = to_mermaid(tree)
    mmd_path = OUT_DIR / "decision_tree.mmd"
    mmd_path.write_text(mmd)
    print(f"Mermaid → {mmd_path.relative_to(OUT_DIR.parent)}")
    print("         (paste at mermaid.live or open in VS Code with Mermaid extension)\n")
    print(mmd)

    if not mermaid_only:
        print()
        save_png(tree, OUT_DIR / "decision_tree.png")
