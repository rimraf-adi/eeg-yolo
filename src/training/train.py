import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np

import sys
import os
# Allow executing gracefully from the root folder or deeply nested execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.yolo1d import yolo_1d_v11_n
from src.training.dataset import EEGRegressionDataset

def calc_metrics(y_true, y_pred, p=18):
    """
    Computes regression metrics across all batched targets.
    p relates to number of input features (channels=18) for Adjusted R2.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean(np.square(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    # Calculate R2
    ss_res = np.sum(np.square(y_true - y_pred), axis=0)
    ss_tot = np.sum(np.square(y_true - np.mean(y_true, axis=0)), axis=0)
    
    r2_per_target = np.ones_like(ss_res)
    nonzero = ss_tot != 0
    r2_per_target[nonzero] = 1 - (ss_res[nonzero] / ss_tot[nonzero])
    r2 = np.mean(r2_per_target)
    
    # Calculate Adjusted R2
    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
    
    return r2, adj_r2, mae, rmse

def train(data_dir, anno_dir, epochs=50, batch_size=16, lr=1e-3, patience=5, results_file="results.txt"):
    # We natively isolated the dataset strictly from P001 to P082!
    EEG_IDX = list(range(1, 83))

    # Split patients into Train / Val / Test (~70-15-15 split)
    random.seed(42) # fixed seed for reproducibility
    shuffled_idx = list(EEG_IDX)
    random.shuffle(shuffled_idx)
    
    n_pids = len(shuffled_idx)
    train_split = int(0.7 * n_pids)
    val_split = int(0.85 * n_pids)
    
    train_pids = shuffled_idx[:train_split]
    val_pids = shuffled_idx[train_split:val_split]
    test_pids = shuffled_idx[val_split:]

    with open(results_file, "w") as f:
        f.write(f"EEG Seizure 1D Multi-Channel YOLO Regression Training\n")
        f.write(f"Total Patients Analyzed: {n_pids}\n")
        f.write(f"Train PIDs: {train_pids}\n")
        f.write(f"Val PIDs: {val_pids}\n")
        f.write(f"Test PIDs: {test_pids}\n")
        f.write("-" * 50 + "\n\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Assigning GPU Engine backend mapping natively: {device}")
    
    print("Loading TRAIN dataset...")
    train_ds = EEGRegressionDataset(data_dir, anno_dir, allowed_pids=train_pids) 
    print("Loading VAL dataset...")
    val_ds = EEGRegressionDataset(data_dir, anno_dir, allowed_pids=val_pids)
    print("Loading TEST dataset...")
    test_ds = EEGRegressionDataset(data_dir, anno_dir, allowed_pids=test_pids)
    
    print(f"Train Windows: {len(train_ds)}, Val Windows: {len(val_ds)}, Test Windows: {len(test_ds)}")
    with open(results_file, "a") as f:
        f.write(f"Train Windows: {len(train_ds)}, Val Windows: {len(val_ds)}, Test Windows: {len(test_ds)}\n\n")

    if len(train_ds) == 0:
        print("Train Dataset empty. Check paths and formats.")
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = yolo_1d_v11_n(in_channels=18).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        train_trues = []
        train_preds = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Train")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Detach for metric computations later
            train_trues.append(y.detach().cpu().numpy())
            train_preds.append(preds.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss /= len(train_loader)
        
        # Calculate train metrics mapping outputs globally
        train_trues = np.vstack(train_trues)
        train_preds = np.vstack(train_preds)
        train_r2, train_adj_r2, train_mae, train_rmse = calc_metrics(train_trues, train_preds)
        
        model.eval()
        val_loss = 0.0
        val_trues = []
        val_preds = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item()
                
                val_trues.append(y.cpu().numpy())
                val_preds.append(preds.cpu().numpy())
                
        if len(val_loader) > 0:
            val_loss /= len(val_loader)
            val_trues = np.vstack(val_trues)
            val_preds = np.vstack(val_preds)
            val_r2, val_adj_r2, val_mae, val_rmse = calc_metrics(val_trues, val_preds)
        else:
            val_r2, val_adj_r2, val_mae, val_rmse = 0.0, 0.0, 0.0, 0.0
        
        log_line = (f"Epoch {epoch}/{epochs}:\n"
                    f"  [Train] Loss={train_loss:.4f} | R²={train_r2:.4f} | Adj-R²={train_adj_r2:.4f} | MAE={train_mae:.4f} | RMSE={train_rmse:.4f}\n"
                    f"  [Val]   Loss={val_loss:.4f}   | R²={val_r2:.4f}   | Adj-R²={val_adj_r2:.4f}   | MAE={val_mae:.4f}   | RMSE={val_rmse:.4f}\n")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pt")
            log_line += "  -> BEST! Saved weights to best_model.pt"
        else:
            epochs_no_improve += 1
            log_line += f"  -> [No improvement: {epochs_no_improve}/{patience}]"

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
        
    print("\n" + "="*50)
    print("Evaluating Test Set with Best Model")
    print("="*50)
    
    try:
        model.load_state_dict(torch.load("best_model.pt", map_location=device, weights_only=True))
    except TypeError:
        # Fallback for older PyTorch versions where weights_only is not supported
        model.load_state_dict(torch.load("best_model.pt", map_location=device))
    except Exception as e:
        print(f"Error loading best_model.pt: {e}")
        return
        
    model.eval()
    
    test_loss = 0.0
    test_trues = []
    test_preds = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            test_loss += loss.item()
            
            test_trues.append(y.cpu().numpy())
            test_preds.append(preds.cpu().numpy())
            
    if len(test_loader) > 0:
        test_loss /= len(test_loader)
        test_trues = np.vstack(test_trues)
        test_preds = np.vstack(test_preds)
        test_r2, test_adj_r2, test_mae, test_rmse = calc_metrics(test_trues, test_preds)
    else:
        test_r2, test_adj_r2, test_mae, test_rmse = 0.0, 0.0, 0.0, 0.0
        
    test_msg = (f"\n--- TEST SET METRICS ---\n"
                f"Loss:   {test_loss:.4f}\n"
                f"R²:     {test_r2:.4f}\n"
                f"Adj-R²: {test_adj_r2:.4f}\n"
                f"MAE:    {test_mae:.4f}\n"
                f"RMSE:   {test_rmse:.4f}\n"
                f"------------------------\n")
    
    print(test_msg)
    with open(results_file, "a") as f:
        f.write(test_msg)

    print(f"Results stored in {results_file} in current directory.")

if __name__ == '__main__':
    train(
        data_dir='/Volumes/WORKSPACE/opensource-dataset/processed/parquet_data',
        anno_dir='/Volumes/WORKSPACE/opensource-dataset/processed/extracted_events',
        epochs=50,
        batch_size=128,
        results_file='results.txt'
    )
