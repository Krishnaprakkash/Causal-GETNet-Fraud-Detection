import pandas as pd
import torch
import numpy as np
from torch_geometric.data import HeteroData
import os
import pickle

def build_test_graph():

    # Load saved train card mapping
    with open("../data/processed/card1_mapping.pkl", "rb") as f:
        card_id_mapping = pickle.load(f)

    df = pd.read_csv(
        "../data/raw/test_transaction.csv",
        usecols=['TransactionID', 'card1', 'TransactionAmt', 'ProductCD']
    )

    df = df.reset_index(drop=True)
    df['tx_node_idx'] = range(len(df))

    # Map using TRAIN mapping
    df['card_node_idx'] = df['card1'].map(card_id_mapping)

    # Drop unseen cards
    df_clean = df.dropna(subset=['card_node_idx'])

    data = HeteroData()

    amt_tensor = torch.tensor(
        np.log1p(df['TransactionAmt'].values),
        dtype=torch.float
    ).view(-1, 1)

    product_dummies = pd.get_dummies(df['ProductCD'], prefix='prod')
    product_tensor = torch.tensor(product_dummies.values, dtype=torch.float)

    data['transaction'].x = torch.cat([amt_tensor, product_tensor], dim=1)
    data['transaction'].num_nodes = len(df)

    # Card nodes count must match TRAIN
    data['card'].num_nodes = len(card_id_mapping)
    data['card'].x = torch.zeros(len(card_id_mapping), 1)

    src = torch.tensor(df_clean['tx_node_idx'].values, dtype=torch.long)
    dst = torch.tensor(df_clean['card_node_idx'].values, dtype=torch.long)

    edge_index = torch.stack([src, dst])
    rev_edge_index = torch.stack([dst, src])

    data['transaction', 'uses', 'card'].edge_index = edge_index
    data['card', 'rev_uses', 'transaction'].edge_index = rev_edge_index

    torch.save(data, '../data/processed/graph_test.pt')
    print("Test graph saved.")

if __name__ == "__main__":
    build_test_graph()