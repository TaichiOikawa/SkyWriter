# Copyright 2026 Taichi Oikawa, Yuki Sano, and Toichi Hiratsuka
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import seaborn as sns

from train_separator import MotionSeparator, MotionDataset


INPUT_FEATURES = ["thumb_x", "thumb_y", "index_x", "index_y", "distance", "dx", "dy", "velocity", "acceleration", "angle", "distance_change"]


# ====== Confusion Matrix Visualization ======
def plot_confusion_matrix_with_metrics(y_true, y_pred, save_path):
    """混同行列とメトリクスを作成して保存"""
    cm = confusion_matrix(y_true, y_pred)

    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = (y_true == y_pred).sum() / len(y_true)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Writing', 'Writing'],
                yticklabels=['Not Writing', 'Writing'],
                ax=ax1)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_title('Confusion Matrix', fontsize=14)

    # Plot metrics bar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [accuracy, precision, recall, f1]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

    bars = ax2.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylim([0, 1.0])
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Evaluation Metrics', fontsize=14)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    ax2.bar_label(bars, fmt='%.4f', padding=3, fontsize=14, label_type='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"混同行列とメトリクス保存完了: {save_path}")
    plt.close()

    return cm


# ====== Evaluation ======
def evaluate(model_path, csv_dir, seq_len=30, batch_size=32):
    """
    既存のモデルとデータを使って評価を実行

    Args:
        model_path: 学習済みモデルのパス (.pt ファイル)
        csv_dir: 評価用データのディレクトリ
        seq_len: シーケンス長
        batch_size: バッチサイズ
    """
    print(f"===== Model Evaluation =====")
    print(f"Model: {model_path}")
    print(f"Data directory: {csv_dir}")
    print(f"Sequence length: {seq_len}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}")

    # モデルのロード
    model = MotionSeparator(input_dim=len(INPUT_FEATURES)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"モデルロード完了")

    # データセットの作成
    dataset = MotionDataset(csv_dir, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"Total samples: {len(dataset)}")

    # 評価
    criterion = nn.BCELoss()
    total_loss = 0
    all_predictions = []
    all_labels = []

    print("\n評価中...")
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = criterion(preds.squeeze(), y)
            total_loss += loss.item()

            # Collect predictions and labels
            predicted = (preds.squeeze() > 0.5).float()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * (all_predictions == all_labels).sum() / len(all_labels)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    # Print results
    print("\n===== Evaluation Results =====")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Print detailed classification report
    print("\n===== Classification Report =====")
    print(classification_report(all_labels, all_predictions,
                                target_names=['Not Writing', 'Writing'],
                                zero_division=0))

    # Plot confusion matrix with metrics
    cm_path = model_path.replace('.pt', '_evaluation_ignore.png')
    cm = plot_confusion_matrix_with_metrics(all_labels, all_predictions, cm_path)

    # Print confusion matrix values
    tn, fp, fn, tp = cm.ravel()
    print("\n===== Confusion Matrix =====")
    print(f"True Negatives (TN):  {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP):  {tp}")

    print("\n評価完了")


if __name__ == "__main__":
    evaluate(
        model_path="./model.pt",
        csv_dir="training_data",
        seq_len=30,
        batch_size=32
    )
