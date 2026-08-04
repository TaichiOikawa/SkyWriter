# Copyright 2026 Taichi Oikawa, Yuki Sano, and Toichi Hiratsuka
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import os
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt


INPUT_FEATURES = ["thumb_x", "thumb_y", "index_x", "index_y", "distance", "dx", "dy"]


# ====== Dataset ======
class MotionDataset(Dataset):
    def __init__(self, csv_dir, seq_len=30):
        self.samples = []
        self.seq_len = seq_len
        files = glob.glob(os.path.join(csv_dir, "*.csv"))

        for f in files:
            df = pd.read_csv(f)
            X = df[INPUT_FEATURES].values
            y = df["is_writing"].values  # 0/1

            # Create sequences
            for i in range(len(X) - seq_len):
                self.samples.append((
                    X[i:i + seq_len],
                    y[i + seq_len - 1]   # label of the last frame in the sequence
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y


# ====== Model ======
class MotionSeparator(nn.Module):
    def __init__(self, input_dim=len(INPUT_FEATURES), hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)          # (B, T, H)
        out = out[:, -1, :]            # (B, H)

        out = self.fc(out)             # (B, 1)
        return torch.sigmoid(out)


# ====== Training ======
def train(csv_dir, model_path="model.pt", seq_len=30, epochs=10, batch_size=32, lr=1e-3, val_split=0.15, test_split=0.15):
    print(f"INPUT_FEATURES: {INPUT_FEATURES}")
    print(f"Training data from: {csv_dir}")
    print(f"Model will be saved to: {model_path}")
    print(f"Seq length: {seq_len}, Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {lr}")
    print(f"Validation split: {val_split}, Test split: {test_split}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}")

    print("===== Training started =====")

    # データセットを訓練/検証/テストに分割
    dataset = MotionDataset(csv_dir, seq_len=seq_len)
    test_size = int(len(dataset) * test_split)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {train_size}, Validation samples: {val_size}, Test samples: {test_size}")

    model = MotionSeparator(input_dim=len(INPUT_FEATURES)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 学習曲線用の損失履歴を記録
    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = criterion(preds.squeeze(), y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                loss = criterion(preds.squeeze(), y)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"モデル保存完了: {model_path}")

    # Test evaluation
    print("\n===== Test Evaluation =====")
    model.eval()
    total_test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = criterion(preds.squeeze(), y)
            total_test_loss += loss.item()

            # Accuracy calculation
            predicted = (preds.squeeze() > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # 学習曲線を出力
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_loss_history, marker='', linewidth=2, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_loss_history, marker='', linewidth=2, label='Validation Loss')
    plt.axhline(y=avg_test_loss, color='r', linestyle='--', linewidth=2, label=f'Test Loss: {avg_test_loss:.4f}')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training, Validation and Test Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # グラフを保存
    curve_path = model_path.replace('.pt', '_loss_curve.png')
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    print(f"学習曲線保存完了: {curve_path}")
    plt.close()


if __name__ == "__main__":
    train("training_data", model_path="model.pt", epochs=200)
