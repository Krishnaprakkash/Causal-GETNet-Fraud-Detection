"""
visualize_graph_3d.py
---------------------
Creates an interactive 3D visualization of a sampled subgraph from the
Heterogeneous Temporal DAG (IEEE-CIS Fraud Detection).

Usage (from PJT2/scripts/):
    python visualize_graph_3d.py

Output:
    PJT2/data/processed/graph_3d.html
"""

import os
import sys
import numpy as np
import networkx as nx
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

GRAPH_PATH = os.path.join(PROCESSED_DIR, "hetero_graph.pt")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "graph_3d.html")

ENTITY_COLS = [
    "card1", "card2", "card3", "card4", "card5", "card6",
    "ProductCD", "P_emaildomain", "addr1", "addr2", "dist1",
]

N_SAMPLE_TRANSACTIONS = 500

# Distinct colors for 11 entity types
ENTITY_COLORS = {
    "card1":         "#FF6B35",
    "card2":         "#F7C59F",
    "card3":         "#EFEFD0",
    "card4":         "#004E89",
    "card5":         "#1A936F",
    "card6":         "#88D498",
    "ProductCD":     "#C6DABF",
    "P_emaildomain": "#9B5DE5",
    "addr1":         "#F15BB5",
    "addr2":         "#FEE440",
    "dist1":         "#00BBF9",
}

FRAUD_COLOR    = "#E63946"   # red
NON_FRAUD_COLOR = "#457B9D"  # blue


# ---------------------------------------------------------------------------
# Step 1: Load graph
# ---------------------------------------------------------------------------

def load_graph():
    """Load the PyG HeteroData object from disk."""
    if not os.path.exists(GRAPH_PATH):
        print(
            f"[ERROR] Graph file not found: {GRAPH_PATH}\n"
            "        Run build_graph.py first to generate hetero_graph.pt."
        )
        sys.exit(1)

    import torch
    print(f"[1/5] Loading graph from {GRAPH_PATH} ...")
    graph = torch.load(GRAPH_PATH, weights_only=False)
    print(f"      Loaded. Node types: {graph.node_types}")
    return graph


# ---------------------------------------------------------------------------
# Step 2: Sample subgraph
# ---------------------------------------------------------------------------

def sample_subgraph(graph):
    """
    Stratified sample of N_SAMPLE_TRANSACTIONS transaction nodes
    (~50% fraud, ~50% non-fraud), then collect all entity nodes
    connected to those transactions.

    Returns
    -------
    sampled_tx_indices : np.ndarray  — original transaction node indices
    entity_node_map    : dict        — {entity_col: np.ndarray of original entity indices}
    edge_map           : dict        — {entity_col: (src_entity_indices, dst_tx_indices)}
                                       indices are into sampled_tx_indices / entity_node_map
    """
    import torch

    print(f"[2/5] Sampling {N_SAMPLE_TRANSACTIONS} transaction nodes (stratified) ...")

    y = graph["transaction"].y.numpy()
    fraud_idx    = np.where(y == 1)[0]
    non_fraud_idx = np.where(y == 0)[0]

    n_fraud    = min(N_SAMPLE_TRANSACTIONS // 2, len(fraud_idx))
    n_non_fraud = N_SAMPLE_TRANSACTIONS - n_fraud
    n_non_fraud = min(n_non_fraud, len(non_fraud_idx))

    rng = np.random.default_rng(seed=42)
    sampled_fraud    = rng.choice(fraud_idx,     size=n_fraud,     replace=False)
    sampled_non_fraud = rng.choice(non_fraud_idx, size=n_non_fraud, replace=False)
    sampled_tx = np.concatenate([sampled_fraud, sampled_non_fraud])

    print(f"      Sampled {n_fraud} fraud + {n_non_fraud} non-fraud transactions.")

    sampled_tx_set = set(sampled_tx.tolist())

    # For each entity type, find edges that touch sampled transactions
    entity_node_map = {}   # entity_col -> array of original entity node indices
    edge_map = {}          # entity_col -> (entity_local_idx, tx_local_idx)

    # Build a local index for sampled transactions
    tx_global_to_local = {g: l for l, g in enumerate(sampled_tx.tolist())}

    for entity_col in ENTITY_COLS:
        edge_type = (entity_col, "uses", "transaction")
        if edge_type not in graph.edge_types:
            continue

        edge_index = graph[edge_type].edge_index.numpy()  # shape (2, E)
        src_entity = edge_index[0]  # entity node indices
        dst_tx     = edge_index[1]  # transaction node indices

        # Keep only edges where dst_tx is in sampled set
        mask = np.isin(dst_tx, sampled_tx)
        if mask.sum() == 0:
            continue

        src_filtered = src_entity[mask]
        dst_filtered = dst_tx[mask]

        # Unique entity nodes involved
        unique_entities = np.unique(src_filtered)
        entity_global_to_local = {g: l for l, g in enumerate(unique_entities.tolist())}

        # Remap to local indices
        src_local = np.array([entity_global_to_local[e] for e in src_filtered])
        dst_local = np.array([tx_global_to_local[t]     for t in dst_filtered])

        entity_node_map[entity_col] = unique_entities
        edge_map[entity_col]        = (src_local, dst_local)

    # Summary
    print(f"      Sampled transaction nodes : {len(sampled_tx)}")
    for ec, arr in entity_node_map.items():
        n_edges = len(edge_map[ec][0])
        print(f"      Entity '{ec}': {len(arr)} nodes, {n_edges} edges")

    return sampled_tx, entity_node_map, edge_map


# ---------------------------------------------------------------------------
# Step 3: Build NetworkX graph
# ---------------------------------------------------------------------------

def build_networkx_graph(graph, sampled_tx, entity_node_map, edge_map):
    """
    Build a NetworkX DiGraph from the sampled subgraph.

    Node ID convention:
        transaction nodes : ("transaction", local_idx)
        entity nodes      : (entity_col, local_idx)
    """
    print("[3/5] Building NetworkX graph ...")

    y = graph["transaction"].y.numpy()
    G = nx.DiGraph()

    # Add transaction nodes
    for local_idx, global_idx in enumerate(sampled_tx):
        node_id = ("transaction", local_idx)
        G.add_node(
            node_id,
            node_type="transaction",
            global_idx=int(global_idx),
            is_fraud=int(y[global_idx]),
        )

    # Add entity nodes and edges
    for entity_col, unique_entities in entity_node_map.items():
        for local_idx, global_idx in enumerate(unique_entities):
            node_id = (entity_col, local_idx)
            G.add_node(
                node_id,
                node_type=entity_col,
                global_idx=int(global_idx),
                is_fraud=-1,
            )

        src_local, dst_local = edge_map[entity_col]
        for s, d in zip(src_local, dst_local):
            src_node = (entity_col, int(s))
            dst_node = ("transaction", int(d))
            G.add_edge(src_node, dst_node, entity_type=entity_col)

    print(f"      NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ---------------------------------------------------------------------------
# Step 4: Compute 3D layout
# ---------------------------------------------------------------------------

def compute_3d_layout(G):
    """Compute 3D spring layout positions for all nodes."""
    print("[4/5] Computing 3D spring layout (this may take a moment) ...")
    pos = nx.spring_layout(G, dim=3, seed=42, k=0.5)
    return pos


# ---------------------------------------------------------------------------
# Step 5: Build Plotly 3D figure
# ---------------------------------------------------------------------------

def build_plotly_figure(G, pos):
    """
    Construct an interactive Plotly 3D scatter figure.

    Returns a plotly.graph_objects.Figure.
    """
    print("[5/5] Building Plotly 3D figure ...")

    # ---- Edge traces (one thin gray line per edge) ----
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="rgba(180,180,180,0.4)", width=1),
        hoverinfo="none",
        name="edges",
        showlegend=False,
    )

    # ---- Node traces grouped by type ----
    node_traces = []

    # Transaction fraud nodes
    tx_fraud_x, tx_fraud_y, tx_fraud_z, tx_fraud_text = [], [], [], []
    tx_non_x,   tx_non_y,   tx_non_z,   tx_non_text   = [], [], [], []

    for node_id, attrs in G.nodes(data=True):
        if attrs["node_type"] != "transaction":
            continue
        x, y, z = pos[node_id]
        hover = (
            f"Type: transaction<br>"
            f"Local idx: {node_id[1]}<br>"
            f"Global idx: {attrs['global_idx']}<br>"
            f"Fraud: {attrs['is_fraud']}"
        )
        if attrs["is_fraud"] == 1:
            tx_fraud_x.append(x); tx_fraud_y.append(y); tx_fraud_z.append(z)
            tx_fraud_text.append(hover)
        else:
            tx_non_x.append(x); tx_non_y.append(y); tx_non_z.append(z)
            tx_non_text.append(hover)

    if tx_fraud_x:
        node_traces.append(go.Scatter3d(
            x=tx_fraud_x, y=tx_fraud_y, z=tx_fraud_z,
            mode="markers",
            marker=dict(size=8, color=FRAUD_COLOR, opacity=0.9,
                        line=dict(width=0.5, color="white")),
            text=tx_fraud_text,
            hoverinfo="text",
            name="Transaction (fraud=1)",
        ))

    if tx_non_x:
        node_traces.append(go.Scatter3d(
            x=tx_non_x, y=tx_non_y, z=tx_non_z,
            mode="markers",
            marker=dict(size=5, color=NON_FRAUD_COLOR, opacity=0.7,
                        line=dict(width=0.5, color="white")),
            text=tx_non_text,
            hoverinfo="text",
            name="Transaction (fraud=0)",
        ))

    # Entity nodes — one trace per entity type
    for entity_col in ENTITY_COLS:
        ex, ey, ez, etxt = [], [], [], []
        for node_id, attrs in G.nodes(data=True):
            if attrs["node_type"] != entity_col:
                continue
            x, y, z = pos[node_id]
            hover = (
                f"Type: {entity_col}<br>"
                f"Local idx: {node_id[1]}<br>"
                f"Global idx: {attrs['global_idx']}"
            )
            ex.append(x); ey.append(y); ez.append(z)
            etxt.append(hover)

        if not ex:
            continue

        node_traces.append(go.Scatter3d(
            x=ex, y=ey, z=ez,
            mode="markers",
            marker=dict(
                size=10,
                color=ENTITY_COLORS.get(entity_col, "#AAAAAA"),
                opacity=0.85,
                symbol="diamond",
                line=dict(width=0.5, color="white"),
            ),
            text=etxt,
            hoverinfo="text",
            name=f"Entity: {entity_col}",
        ))

    # ---- Assemble figure ----
    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            title=dict(
                text="Heterogeneous Temporal DAG — Sampled Subgraph (3D)",
                font=dict(size=18),
            ),
            showlegend=True,
            legend=dict(
                itemsizing="constant",
                font=dict(size=11),
            ),
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False, title=""),
                yaxis=dict(showbackground=False, showticklabels=False, title=""),
                zaxis=dict(showbackground=False, showticklabels=False, title=""),
                bgcolor="rgb(15,15,25)",
            ),
            paper_bgcolor="rgb(15,15,25)",
            font=dict(color="white"),
            margin=dict(l=0, r=0, t=50, b=0),
            hovermode="closest",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Step 6: Print summary
# ---------------------------------------------------------------------------

def print_summary(G, sampled_tx, entity_node_map, edge_map):
    """Print node/edge counts for the sampled subgraph."""
    print("\n" + "=" * 60)
    print("  Sampled Subgraph Summary")
    print("=" * 60)

    y_vals = [d["is_fraud"] for _, d in G.nodes(data=True) if d["node_type"] == "transaction"]
    n_fraud    = sum(1 for v in y_vals if v == 1)
    n_non_fraud = sum(1 for v in y_vals if v == 0)
    print(f"  transaction nodes : {len(y_vals):>6}  (fraud={n_fraud}, non-fraud={n_non_fraud})")

    for entity_col in ENTITY_COLS:
        if entity_col not in entity_node_map:
            continue
        n_nodes = len(entity_node_map[entity_col])
        n_edges = len(edge_map[entity_col][0])
        print(f"  {entity_col:<16} nodes : {n_nodes:>6}  edges : {n_edges:>6}")

    print(f"\n  Total nodes : {G.number_of_nodes()}")
    print(f"  Total edges : {G.number_of_edges()}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    graph = load_graph()
    sampled_tx, entity_node_map, edge_map = sample_subgraph(graph)
    G = build_networkx_graph(graph, sampled_tx, entity_node_map, edge_map)
    pos = compute_3d_layout(G)
    fig = build_plotly_figure(G, pos)

    print_summary(G, sampled_tx, entity_node_map, edge_map)

    fig.write_html(OUTPUT_PATH)
    print(f"\n[Done] Interactive 3D visualization saved to:\n       {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
