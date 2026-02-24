import pandas as pd
import torch
import numpy as np
from torch_geometric.data import HeteroData
import os

def build_bipartite_graph():
    print("1. Loading Data...")
    df = pd.read_csv('../data/raw/train_transaction.csv', 
                     usecols=['TransactionID', 'isFraud', 'card1', 'TransactionAmt', 'ProductCD'])
    
    print(f"   Loaded {len(df)} transactions.")

    # ---------------------------------------------------------
    # STEP 1: Process 'Team B' (Cards)
    # ---------------------------------------------------------
    print("2. Processing Card Nodes...")
    unique_cards = df['card1'].dropna().unique()
    card_id_mapping = {real_id: idx for idx, real_id in enumerate(unique_cards)}
    print(f"   Found {len(unique_cards)} unique cards.")

    # ---------------------------------------------------------
    # STEP 2: Process 'Team A' (Transactions)
    # ---------------------------------------------------------
    print("3. Processing Transaction Nodes & Features...")
    df['tx_node_idx'] = range(len(df))
    df['card_node_idx'] = df['card1'].map(card_id_mapping)
    df_clean = df.dropna(subset=['card_node_idx'])

    # Feature Engineering
    amt_tensor = torch.tensor(np.log1p(df['TransactionAmt'].values), dtype=torch.float).view(-1, 1)
    product_dummies = pd.get_dummies(df['ProductCD'], prefix='prod')
    product_tensor = torch.tensor(product_dummies.values, dtype=torch.float)
    tx_features = torch.cat([amt_tensor, product_tensor], dim=1)

    # ---------------------------------------------------------
    # STEP 3: Build Forward Connections
    # ---------------------------------------------------------
    print("4. Building Connections...")
    src = torch.tensor(df_clean['tx_node_idx'].values, dtype=torch.long)
    dst = torch.tensor(df_clean['card_node_idx'].values, dtype=torch.long)
    
    # Edge: Transaction -> Card
    edge_index = torch.stack([src, dst], dim=0)
    
    # NEW: Edge: Card -> Transaction (Reverse)
    # We simply flip the source and destination
    rev_edge_index = torch.stack([dst, src], dim=0)

    # ---------------------------------------------------------
    # STEP 4: Assemble Graph
    # ---------------------------------------------------------
    print("5. Assembling HeteroData Graph...")
    data = HeteroData()

    # Nodes
    data['transaction'].num_nodes = len(df)
    data['transaction'].y = torch.tensor(df['isFraud'].values, dtype=torch.float)
    data['transaction'].x = tx_features
    data['card'].num_nodes = len(unique_cards)
    
    # We need to give Cards some dummy features so the math works 
    # (The AI expects x for all node types involved in message passing)
    data['card'].x = torch.zeros(len(unique_cards), 1) 

    # Edges (Forward AND Backward)
    data['transaction', 'uses', 'card'].edge_index = edge_index
    data['card', 'rev_uses', 'transaction'].edge_index = rev_edge_index

    print("\nGraph Construction Complete!")
    print(data)

    os.makedirs('../data/processed', exist_ok=True)
    torch.save(data, '../data/processed/graph_data.pt')
    print("Graph saved to 'data/processed/graph_data.pt'")

if __name__ == "__main__":
    build_bipartite_graph()