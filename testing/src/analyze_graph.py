import torch
import numpy as np
from collections import Counter
from torch_geometric.utils import to_undirected
import networkx as nx

def analyze():

    data = torch.load("../data/processed/graph_data.pt", weights_only=False)

    # ------------------------------------------------
    # Basic info
    # ------------------------------------------------
    num_tx = data["transaction"].num_nodes
    num_card = data["card"].num_nodes
    labels = data["transaction"].y.numpy()

    print("Transactions:", num_tx)
    print("Card nodes:", num_card)
    print("Fraud rate:", labels.mean())

    # ------------------------------------------------
    # Build undirected edge index for connectivity
    # ------------------------------------------------
    edge_index = data["transaction", "uses", "card"].edge_index

    # Shift card indices so both node types share one space
    card_offset = num_tx
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy() + card_offset

    full_edges = np.vstack([src, dst])
    full_edges = np.hstack([full_edges, full_edges[::-1]])

    G = nx.Graph()
    G.add_edges_from(full_edges.T)

    # ------------------------------------------------
    # 1️⃣ Connectivity Patterns
    # ------------------------------------------------
    components = list(nx.connected_components(G))
    comp_sizes = [len(c) for c in components]

    print("\nConnected Components:", len(components))
    print("Largest component size:", max(comp_sizes))

    # Degree per card
    card_degrees = edge_index[1].numpy()
    card_degree_count = Counter(card_degrees)

    print("\nAverage transactions per card:",
          np.mean(list(card_degree_count.values())))

    print("Top 5 high-degree cards:",
          sorted(card_degree_count.values(), reverse=True)[:5])

    # ------------------------------------------------
    # 2️⃣ Fraud Clustering per Card
    # ------------------------------------------------
    fraud_per_card = {}

    for tx_idx, card_idx in zip(edge_index[0].numpy(),
                                edge_index[1].numpy()):

        if card_idx not in fraud_per_card:
            fraud_per_card[card_idx] = []

        fraud_per_card[card_idx].append(labels[tx_idx])

    fraud_ratios = [
        np.mean(v) for v in fraud_per_card.values()
        if len(v) > 1
    ]

    print("\nAverage fraud ratio per card:",
          np.mean(fraud_ratios))

    print("Max fraud ratio on a card:",
          np.max(fraud_ratios))

    # ------------------------------------------------
    # 3️⃣ Degree Distribution Fraud vs Non-Fraud
    # ------------------------------------------------
    tx_degree = Counter(edge_index[0].numpy())

    fraud_degrees = []
    nonfraud_degrees = []

    for tx_idx, deg in tx_degree.items():
        if labels[tx_idx] == 1:
            fraud_degrees.append(deg)
        else:
            nonfraud_degrees.append(deg)

    print("\nAverage degree (Fraud tx):",
          np.mean(fraud_degrees))

    print("Average degree (Non-fraud tx):",
          np.mean(nonfraud_degrees))


if __name__ == "__main__":
    analyze()