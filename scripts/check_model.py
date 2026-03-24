import os
import torch

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "best_model.pt")

checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
print("=" * 50)
print("BEST MODEL PARAMETERS")
print("=" * 50)
print(f"Epoch: {checkpoint['epoch']}")
print(f"Validation AUC: {checkpoint['val_auc']:.4f}")
print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
print(f"Training AUC: {checkpoint['train_auc']:.4f}")
print(f"Training Loss: {checkpoint['train_loss']:.4f}")
print("=" * 50)
print("\nModel Layers:")
for key, val in checkpoint['model_state_dict'].items():
    print(f"  {key}: {list(val.shape)}")