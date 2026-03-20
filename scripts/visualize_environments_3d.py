"""
visualize_environments_3d.py
-----------------------------
Creates an interactive 3D visualization of the graph with environment coloring.
Shows three separate visualizations for env_time, env_region, and env_fraud_rate.

Usage (from PJT2/scripts/):
    python visualize_environments_3d.py

Output:
    PJT2/data/processed/environments_3d.html
"""

import os
import sys
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

GRAPH_PATH = os.path.join(PROCESSED_DIR, "hetero_graph_with_env.pt")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "environments_3d.html")

ENTITY_COLS = [
    "card1", "card2", "card3", "card4", "card5", "card6",
    "ProductCD", "P_emaildomain", "addr1", "addr2", "dist1",
]

# Environment-specific sampling
N_SAMPLE_PER_ENV_TIME = 200       # 8 time windows * 200 = 1600
N_SAMPLE_PER_ENV_REGION = 200     # 5 region bins * 200 = 1000
N_SAMPLE_PER_ENV_FRAUD_RATE = 300 # 3 fraud rate regimes * 300 = 900

# Environment colors
ENV_TIME_COLORS = [
    "#E63946", "#F4A261", "#E9C46A", "#2A9D8F",
    "#264653", "#9B5DE5", "#00F5D4", "#FF6B6B",
]  # 8 colors for time bins

ENV_REGION_COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#F4A261", "#9B5DE5",
]  # 5 colors for regions

ENV_FRAUD_RATE_COLORS = [
    "#2A9D8F", "#F4A261", "#E63946",
]  # low, medium, high fraud rate (green, orange, red)

ENTITY_COLOR = "#888888"  # gray for entity nodes
FRAUD_MARKER = "diamond"
NON_FRAUD_MARKER = "circle"


# ---------------------------------------------------------------------------
# Step 1: Load graph with environment labels
# ---------------------------------------------------------------------------

def load_graph():
    """Load the PyG HeteroData object with environment labels from disk."""
    if not os.path.exists(GRAPH_PATH):
        print(
            f"[ERROR] Graph file not found: {GRAPH_PATH}\n"
            "        Run assign_environments.py first to generate hetero_graph_with_env.pt."
        )
        sys.exit(1)

    import torch
    print(f"[1/6] Loading graph from {GRAPH_PATH} ...")
    graph = torch.load(GRAPH_PATH, weights_only=False)
    print(f"      Loaded. Node types: {graph.node_types}")
    
    # Verify environment attributes exist
    if not hasattr(graph["transaction"], "env_time"):
        print("[ERROR] Graph missing env_time attribute. Run assign_environments.py first.")
        sys.exit(1)
    if not hasattr(graph["transaction"], "env_region"):
        print("[ERROR] Graph missing env_region attribute. Run assign_environments.py first.")
        sys.exit(1)
    if not hasattr(graph["transaction"], "env_fraud_rate"):
        print("[ERROR] Graph missing env_fraud_rate attribute. Run assign_environments.py first.")
        sys.exit(1)
    
    print(f"      Environment attributes found:")
    print(f"        - env_time: {graph['transaction'].env_time.max().item() + 1} bins (0-{graph['transaction'].env_time.max().item()})")
    print(f"        - env_region: {graph['transaction'].env_region.max().item() + 1} bins (0-{graph['transaction'].env_region.max().item()})")
    print(f"        - env_fraud_rate: {graph['transaction'].env_fraud_rate.max().item() + 1} bins (0-{graph['transaction'].env_fraud_rate.max().item()})")
    
    return graph


# ---------------------------------------------------------------------------
# Step 2: Sample subgraph by environment
# ---------------------------------------------------------------------------

def sample_by_environment(graph, env_attr, n_per_bin):
    """
    Sample transaction nodes by environment bin.
    
    Parameters
    ----------
    graph : HeteroData
        The graph with environment attributes
    env_attr : str
        Name of environment attribute ('env_time', 'env_region', 'env_fraud_rate')
    n_per_bin : int
        Number of transactions to sample per environment bin
        
    Returns
    -------
    sampled_tx : np.ndarray
        Global indices of sampled transaction nodes
    env_labels : np.ndarray
        Environment labels for each sampled transaction
    """
    import torch
    
    print(f"\n[2/6] Sampling by {env_attr} ({n_per_bin} per bin) ...")
    
    env_vals = getattr(graph["transaction"], env_attr).numpy()
    y = graph["transaction"].y.numpy()
    
    unique_bins = np.unique(env_vals)
    print(f"      Found {len(unique_bins)} bins: {unique_bins}")
    
    sampled_tx = []
    sampled_env_labels = []
    
    rng = np.random.default_rng(seed=42)
    
    for bin_idx in unique_bins:
        bin_mask = env_vals == bin_idx
        bin_indices = np.where(bin_mask)[0]
        
        # Sample up to n_per_bin from this bin
        n_available = len(bin_indices)
        n_to_sample = min(n_per_bin, n_available)
        
        if n_to_sample > 0:
            sampled = rng.choice(bin_indices, size=n_to_sample, replace=False)
            sampled_tx.extend(sampled.tolist())
            sampled_env_labels.extend([bin_idx] * n_to_sample)
            
            # Calculate fraud rate for this bin
            bin_fraud_rate = y[sampled].mean()
            print(f"      Bin {bin_idx}: sampled {n_to_sample}, fraud rate = {bin_fraud_rate:.3f}")
    
    sampled_tx = np.array(sampled_tx)
    sampled_env_labels = np.array(sampled_env_labels)
    
    print(f"      Total sampled: {len(sampled_tx)} transactions")
    
    return sampled_tx, sampled_env_labels


def collect_connected_entities(graph, sampled_tx):
    """
    Collect all entity nodes connected to the sampled transactions.
    
    Returns
    -------
    entity_node_map : dict
        {entity_col: np.ndarray of original entity indices}
    edge_map : dict
        {entity_col: (src_entity_indices, dst_tx_indices)}
        indices are local indices into sampled_tx / entity_node_map
    """
    import torch
    
    sampled_tx_set = set(sampled_tx.tolist())
    tx_global_to_local = {g: l for l, g in enumerate(sampled_tx.tolist())}
    
    entity_node_map = {}
    edge_map = {}
    
    for entity_col in ENTITY_COLS:
        edge_type = (entity_col, "uses", "transaction")
        if edge_type not in graph.edge_types:
            continue
        
        edge_index = graph[edge_type].edge_index.numpy()
        src_entity = edge_index[0]
        dst_tx = edge_index[1]
        
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
        dst_local = np.array([tx_global_to_local[t] for t in dst_filtered])
        
        entity_node_map[entity_col] = unique_entities
        edge_map[entity_col] = (src_local, dst_local)
    
    return entity_node_map, edge_map


# ---------------------------------------------------------------------------
# Step 3: Build NetworkX graph
# ---------------------------------------------------------------------------

def build_networkx_graph(graph, sampled_tx, env_labels, env_attr, entity_node_map, edge_map):
    """
    Build a NetworkX DiGraph from the sampled subgraph with environment labels.
    """
    print(f"[3/6] Building NetworkX graph with {env_attr} labels ...")
    
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
            env_label=int(env_labels[local_idx]),
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
                env_label=-1,
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
    print("[4/6] Computing 3D spring layout (this may take a moment) ...")
    pos = nx.spring_layout(G, dim=3, seed=42, k=0.5)
    return pos


# ---------------------------------------------------------------------------
# Step 5: Build Plotly 3D figure
# ---------------------------------------------------------------------------

def build_plotly_figure(G, pos, env_attr, env_colors, title_suffix=""):
    """
    Construct an interactive Plotly 3D scatter figure with environment coloring.
    """
    print(f"[5/6] Building Plotly 3D figure for {env_attr} ...")
    
    # ---- Edge traces ----
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
        line=dict(color="rgba(180,180,180,0.3)", width=1),
        hoverinfo="none",
        name="edges",
        showlegend=False,
    )
    
    # ---- Node traces grouped by environment and fraud ----
    node_traces = []
    
    # Group transaction nodes by environment label
    env_bins = {}
    for node_id, attrs in G.nodes(data=True):
        if attrs["node_type"] != "transaction":
            continue
        env_label = attrs["env_label"]
        if env_label not in env_bins:
            env_bins[env_label] = {"fraud": [], "non_fraud": []}
        x, y, z = pos[node_id]
        if attrs["is_fraud"] == 1:
            env_bins[env_label]["fraud"].append((x, y, z, node_id, attrs))
        else:
            env_bins[env_label]["non_fraud"].append((x, y, z, node_id, attrs))
    
    # Create traces for each environment bin
    for env_label in sorted(env_bins.keys()):
        color = env_colors[env_label] if env_label < len(env_colors) else "#888888"
        
        # Fraud nodes in this environment
        fraud_data = env_bins[env_label]["fraud"]
        if fraud_data:
            fx, fy, fz, ftext = [], [], [], []
            for x, y, z, node_id, attrs in fraud_data:
                fx.append(x); fy.append(y); fz.append(z)
                hover = (
                    f"Type: transaction<br>"
                    f"Local idx: {node_id[1]}<br>"
                    f"Global idx: {attrs['global_idx']}<br>"
                    f"Fraud: {attrs['is_fraud']}<br>"
                    f"{env_attr}: {attrs['env_label']}"
                )
                ftext.append(hover)
            
            node_traces.append(go.Scatter3d(
                x=fx, y=fy, z=fz,
                mode="markers",
                marker=dict(
                    size=8, 
                    color=color, 
                    opacity=0.9,
                    symbol="diamond",
                    line=dict(width=0.5, color="white"),
                ),
                text=ftext,
                hoverinfo="text",
                name=f"Env {env_label} (fraud)",
            ))
        
        # Non-fraud nodes in this environment
        non_fraud_data = env_bins[env_label]["non_fraud"]
        if non_fraud_data:
            nx, ny, nz, ntext = [], [], [], []
            for x, y, z, node_id, attrs in non_fraud_data:
                nx.append(x); ny.append(y); nz.append(z)
                hover = (
                    f"Type: transaction<br>"
                    f"Local idx: {node_id[1]}<br>"
                    f"Global idx: {attrs['global_idx']}<br>"
                    f"Fraud: {attrs['is_fraud']}<br>"
                    f"{env_attr}: {attrs['env_label']}"
                )
                ntext.append(hover)
            
            node_traces.append(go.Scatter3d(
                x=nx, y=ny, z=nz,
                mode="markers",
                marker=dict(
                    size=5, 
                    color=color, 
                    opacity=0.7,
                    symbol="circle",
                    line=dict(width=0.5, color="white"),
                ),
                text=ntext,
                hoverinfo="text",
                name=f"Env {env_label} (non-fraud)",
            ))
    
    # ---- Entity nodes (gray diamonds) ----
    ex, ey, ez, etxt = [], [], [], []
    for node_id, attrs in G.nodes(data=True):
        if attrs["node_type"] == "transaction":
            continue
        x, y, z = pos[node_id]
        ex.append(x); ey.append(y); ez.append(z)
        hover = (
            f"Type: {attrs['node_type']}<br>"
            f"Local idx: {node_id[1]}<br>"
            f"Global idx: {attrs['global_idx']}"
        )
        etxt.append(hover)
    
    if ex:
        node_traces.append(go.Scatter3d(
            x=ex, y=ey, z=ez,
            mode="markers",
            marker=dict(
                size=10,
                color=ENTITY_COLOR,
                opacity=0.6,
                symbol="diamond",
                line=dict(width=0.5, color="white"),
            ),
            text=etxt,
            hoverinfo="text",
            name="Entity nodes",
        ))
    
    # ---- Assemble figure ----
    title = f"Heterogeneous Temporal DAG — {env_attr} Coloring (3D){title_suffix}"
    
    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            title=dict(
                text=title,
                font=dict(size=18),
                x=0.5,
            ),
            showlegend=True,
            legend=dict(
                itemsizing="constant",
                font=dict(size=11),
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
            ),
            autosize=True,
            margin=dict(l=0, r=0, t=40, b=0),
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False, title=""),
                yaxis=dict(showbackground=False, showticklabels=False, title=""),
                zaxis=dict(showbackground=False, showticklabels=False, title=""),
                bgcolor="rgb(15,15,25)",
                aspectmode="cube",
            ),
            paper_bgcolor="rgb(15,15,25)",
            font=dict(color="white"),
            hovermode="closest",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Step 6: Print environment statistics
# ---------------------------------------------------------------------------

def print_environment_stats(graph, env_attr, sampled_tx, env_labels):
    """Print statistics for the sampled environment."""
    print(f"\n{'=' * 60}")
    print(f"  {env_attr} Statistics")
    print(f"{'=' * 60}")
    
    y = graph["transaction"].y.numpy()
    unique_bins = np.unique(env_labels)
    
    for bin_idx in sorted(unique_bins):
        mask = env_labels == bin_idx
        n_samples = mask.sum()
        fraud_count = y[sampled_tx[mask]].sum()
        fraud_rate = y[sampled_tx[mask]].mean()
        
        print(f"  {env_attr} bin {bin_idx}: n={n_samples:4d}, fraud={int(fraud_count):3d}, fraud_rate={fraud_rate:.3f}")
    
    print(f"{'=' * 60}")


def create_combined_html(figures, titles):
    """Create a single HTML file with all figures."""
    print("\n[6/6] Creating combined HTML with tabs ...")
    
    html_parts = []
    
    # HTML header with CSS for tabs
    html_parts.append("""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Environment Visualizations (3D)</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {
            box-sizing: border-box;
        }
        html, body {
            width: 100%;
            height: 100%;
            margin: 0;
            padding: 0;
            overflow: hidden;
        }
        body {
            font-family: Arial, sans-serif;
            background-color: #1a1a2e;
            color: white;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        h1 {
            text-align: center;
            margin: 10px 0;
            font-size: 20px;
            flex-shrink: 0;
        }
        .tab-container {
            display: flex;
            justify-content: center;
            margin-bottom: 10px;
            flex-shrink: 0;
        }
        .tab {
            padding: 10px 20px;
            margin: 0 5px;
            background-color: #16213e;
            border: 1px solid #0f3460;
            color: white;
            cursor: pointer;
            border-radius: 5px 5px 0 0;
            transition: background-color 0.3s;
        }
        .tab:hover {
            background-color: #0f3460;
        }
        .tab.active {
            background-color: #0f3460;
            border-bottom: 2px solid #e94560;
        }
        .figure-container {
            display: none;
            flex: 1;
            min-height: 0;
            min-width: 0;
            padding: 0 10px 10px 10px;
        }
        .figure-container.active {
            display: flex;
        }
        .figure-container > div {
            flex: 1;
            min-width: 0;
            display: flex;
        }
        .figure-container .plotly-graph-div {
            width: 100% !important;
            height: 100% !important;
            min-width: 0 !important;
        }
        .stats {
            background-color: #16213e;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            flex-shrink: 0;
            max-height: 150px;
            overflow-y: auto;
        }
        .stats h3 {
            margin-top: 0;
            color: #e94560;
            font-size: 14px;
        }
        .stats p {
            font-size: 12px;
            margin: 3px 0;
        }
    </style>
</head>
<body>
    <h1>Heterogeneous Temporal DAG — Environment Visualizations (3D)</h1>
    
    <div class="tab-container">
""")
    
    # Add tab buttons
    for i, title in enumerate(titles):
        active = " active" if i == 0 else ""
        html_parts.append(f'        <button class="tab{active}" onclick="showFigure({i})">{title}</button>\n')
    
    html_parts.append("""    </div>
""")
    
    # Add JavaScript for tab switching
    html_parts.append("""
    <script>
        function showFigure(index) {
            // Hide all figures
            document.querySelectorAll('.figure-container').forEach(function(el) {
                el.classList.remove('active');
            });
            // Show selected figure
            document.getElementById('figure-' + index).classList.add('active');
            // Update tab buttons
            document.querySelectorAll('.tab').forEach(function(el, i) {
                if (i === index) {
                    el.classList.add('active');
                } else {
                    el.classList.remove('active');
                }
            });
            // Trigger resize to fix Plotly chart sizing
            setTimeout(function() {
                window.dispatchEvent(new Event('resize'));
            }, 100);
        }
        // Initial resize on load
        window.addEventListener('load', function() {
            setTimeout(function() {
                window.dispatchEvent(new Event('resize'));
            }, 500);
        });
    </script>
""")
    
    # Add figures
    for i, (fig, title) in enumerate(zip(figures, titles)):
        active = " active" if i == 0 else ""
        html_parts.append(f'    <div id="figure-{i}" class="figure-container{active}">\n')
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
        html_parts.append("    </div>\n")
    
    # Close HTML
    html_parts.append("</body></html>")
    
    return "".join(html_parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Load graph
    graph = load_graph()
    
    figures = []
    titles = []
    
    # -------------------------------------------------------------------------
    # Visualization A: env_time (8 bins)
    # -------------------------------------------------------------------------
    sampled_tx_time, env_labels_time = sample_by_environment(
        graph, "env_time", N_SAMPLE_PER_ENV_TIME
    )
    entity_node_map_time, edge_map_time = collect_connected_entities(graph, sampled_tx_time)
    G_time = build_networkx_graph(
        graph, sampled_tx_time, env_labels_time, "env_time",
        entity_node_map_time, edge_map_time
    )
    pos_time = compute_3d_layout(G_time)
    fig_time = build_plotly_figure(
        G_time, pos_time, "env_time", ENV_TIME_COLORS, 
        title_suffix="<br><sub>8 time windows (earlier → later)</sub>"
    )
    figures.append(fig_time)
    titles.append("Time Windows (env_time)")
    print_environment_stats(graph, "env_time", sampled_tx_time, env_labels_time)
    
    # -------------------------------------------------------------------------
    # Visualization B: env_region (5 bins)
    # -------------------------------------------------------------------------
    sampled_tx_region, env_labels_region = sample_by_environment(
        graph, "env_region", N_SAMPLE_PER_ENV_REGION
    )
    entity_node_map_region, edge_map_region = collect_connected_entities(graph, sampled_tx_region)
    G_region = build_networkx_graph(
        graph, sampled_tx_region, env_labels_region, "env_region",
        entity_node_map_region, edge_map_region
    )
    pos_region = compute_3d_layout(G_region)
    fig_region = build_plotly_figure(
        G_region, pos_region, "env_region", ENV_REGION_COLORS,
        title_suffix="<br><sub>5 region bins (by addr1)</sub>"
    )
    figures.append(fig_region)
    titles.append("Regions (env_region)")
    print_environment_stats(graph, "env_region", sampled_tx_region, env_labels_region)
    
    # -------------------------------------------------------------------------
    # Visualization C: env_fraud_rate (3 bins)
    # -------------------------------------------------------------------------
    sampled_tx_fraud, env_labels_fraud = sample_by_environment(
        graph, "env_fraud_rate", N_SAMPLE_PER_ENV_FRAUD_RATE
    )
    entity_node_map_fraud, edge_map_fraud = collect_connected_entities(graph, sampled_tx_fraud)
    G_fraud = build_networkx_graph(
        graph, sampled_tx_fraud, env_labels_fraud, "env_fraud_rate",
        entity_node_map_fraud, edge_map_fraud
    )
    pos_fraud = compute_3d_layout(G_fraud)
    fig_fraud = build_plotly_figure(
        G_fraud, pos_fraud, "env_fraud_rate", ENV_FRAUD_RATE_COLORS,
        title_suffix="<br><sub>3 fraud-rate regimes (low → medium → high)</sub>"
    )
    figures.append(fig_fraud)
    titles.append("Fraud Rate Regimes (env_fraud_rate)")
    print_environment_stats(graph, "env_fraud_rate", sampled_tx_fraud, env_labels_fraud)
    
    # -------------------------------------------------------------------------
    # Create combined HTML with tabs
    # -------------------------------------------------------------------------
    combined_html = create_combined_html(figures, titles)
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(combined_html)
    
    print(f"\n[Done] Interactive 3D visualization saved to:")
    print(f"       {OUTPUT_PATH}")
    print(f"\nOpen this file in a browser to view the visualizations.")


if __name__ == "__main__":
    main()
