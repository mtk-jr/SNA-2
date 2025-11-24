# src/train_gnn.py
import os
import random
import argparse
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch_geometric.loader import DataLoader

from gnn_dataset import generate_dataset_from_graph
from gnn_model import SimpleGCN
import networkx as nx

# You will need to import your build_network loader
from build_network import load_network_from_csv

def train_epoch(model, loader, optim, device):
    model.train()
    total_loss = 0.0
    criterion = torch.nn.BCEWithLogitsLoss()
    for batch in loader:
        batch = batch.to(device)
        optim.zero_grad()
        logits = model(batch.x, batch.edge_index)  # shape [N_total]
        # Flatten: node-level labels
        y = batch.y.float()
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).long().cpu().numpy()
        ys.append(batch.y.cpu().numpy())
        preds.append(pred)
    if not ys:
        return {}
    y = np.concatenate(ys)
    p = np.concatenate(preds)
    return {
        'accuracy': accuracy_score(y, p),
        'f1': f1_score(y, p, zero_division=0),
        'precision': precision_score(y, p, zero_division=0),
        'recall': recall_score(y, p, zero_division=0)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV (sample_interbank_network.csv)")
    parser.add_argument("--samples", type=int, default=300, help="Number of training samples (shock scenarios)")
    parser.add_argument("--k", type=int, default=1, help="Initial failed banks per sample")
    parser.add_argument("--attack", type=str, default="random", help="attack mode: random/top_degree/top_pagerank")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load network
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = os.path.join("data", "sample_interbank_network.csv")
    print("Loading:", csv_path)
    G = load_network_from_csv(csv_path)

    # generate dataset: many Data objects (same graph, diff initial shock)
    data_list, scaler = generate_dataset_from_graph(
        G,
        num_samples=args.samples,
        initial_k=args.k,
        threshold=args.threshold,
        attack_mode=args.attack,
        random_seed=args.seed
    )

    # split
    random.shuffle(data_list)
    train_n = int(0.8 * len(data_list))
    train_data = data_list[:train_n]
    test_data = data_list[train_n:]

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    in_ch = train_data[0].x.shape[1]
    model = SimpleGCN(in_ch, hidden_channels=64, dropout=0.5).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_f1 = 0.0
    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, train_loader, optim, device)
        metrics = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:03d} loss={loss:.4f} val_acc={metrics.get('accuracy',0):.3f} f1={metrics.get('f1',0):.3f}")
        if metrics.get('f1', 0) > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), "best_gcn_model.pt")
    print("Training complete. Best F1:", best_f1)

    # final evaluation
    metrics = evaluate(model, test_loader, device)
    print("FINAL METRICS:", metrics)

    # Save scaler for future inference
    import joblib
    joblib.dump(scaler, "feature_scaler.pkl")
    print("Saved scaler and model.")

if __name__ == "__main__":
    main()
