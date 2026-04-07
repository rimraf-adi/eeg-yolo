import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

from yolo1d import yolo_1d_v11_n
from dataset import EEGRegressionDataset

def train(data_dir, anno_dir, epochs=50, batch_size=16, lr=1e-3, patience=5, results_file="results.txt"):
    EEG_IDX = [1, 4, 5, 7, 9, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 25, 31, 34,
               36, 38, 39, 40, 41, 44, 47, 50, 51, 52, 62, 63, 66, 67, 69, 73, 75,
               76, 77, 78, 79]

    # Split patients into Train / Test (80-20 split)
    random.seed(42) # fixed seed for reproducibility
    shuffled_idx = list(EEG_IDX)
    random.shuffle(shuffled_idx)
    
    split_point = int(0.8 * len(shuffled_idx))
    train_pids = shuffled_idx[:split_point]
    test_pids = shuffled_idx[split_point:]

    with open(results_file, "w") as f:
        f.write(f"EEG Seizure 1D Regression Training\n")
        f.write(f"Total Patients (from EEG_IDX): {len(EEG_IDX)}\n")
        f.write(f"Train PIDs: {train_pids}\n")
        f.write(f"Test PIDs (Validation): {test_pids}\n")
        f.write("-" * 50 + "\n\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading TRAIN dataset...")
    train_ds = EEGRegressionDataset(data_dir, anno_dir, allowed_pids=train_pids) 
    print("Loading TEST dataset...")
    val_ds = EEGRegressionDataset(data_dir, anno_dir, allowed_pids=test_pids)
    
    print(f"Train Windows: {len(train_ds)}, Validation Windows: {len(val_ds)}")
    with open(results_file, "a") as f:
        f.write(f"Train Windows: {len(train_ds)}, Validation Windows: {len(val_ds)}\n\n")

    if len(train_ds) == 0:
        print("Train Dataset empty. Check paths and formats.")
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = yolo_1d_v11_n(in_channels=18).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Train")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item()
                
        if len(val_loader) > 0:
            val_loss /= len(val_loader)
        
        log_line = f"Epoch {epoch}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pt")
            log_line += " -> BEST! Saved weights to best_model.pt"
        else:
            epochs_no_improve += 1
            log_line += f" -> [No improvement: {epochs_no_improve}/{patience}]"

        print(log_line)
        with open(results_file, "a") as f:
            f.write(log_line + "\n")
            
        if epochs_no_improve >= patience:
            msg = f"\nEarly stopping triggered. Best Validation Loss: {best_val_loss:.4f}."
            print(msg)
            with open(results_file, "a") as f:
                f.write(msg + "\n")
            break

    with open(results_file, "a") as f:
        f.write(f"\nTraining completed.\n")
    print(f"Results stored in {results_file} in current directory.")

if __name__ == '__main__':
    train(
        data_dir='/Volumes/WORKSPACE/neonatal',
        anno_dir='/Volumes/WORKSPACE/dense_annotations_neonatal',
        epochs=50,
        batch_size=8,
        results_file='results.txt'
    )
