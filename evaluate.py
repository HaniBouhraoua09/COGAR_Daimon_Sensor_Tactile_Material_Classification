"""
evaluate.py - Load all trained checkpoints, compute metrics, generate plots.

Usage:
    python evaluate.py
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_loader import get_dataloaders
from models import build_model
from train import ALL_MODELS


def predict(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(batch).argmax(dim=1).cpu().numpy()
            all_pred.extend(pred)
            all_true.extend(batch['label'].cpu().numpy())
    return np.array(all_true), np.array(all_pred)


def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, label_map = get_dataloaders(batch_size=8)
    classes = sorted(label_map.keys(), key=lambda k: label_map[k])
    num_classes = len(classes)
    
    print(f"Classes: {classes}")
    print(f"Device:  {device}")
    
    # ----- Confusion matrices for each model -----
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    results = []
    for i, name in enumerate(ALL_MODELS):
        ckpt_path = f"checkpoints/{name}_best.pt"
        if not os.path.exists(ckpt_path):
            print(f"  [skip] {name}: no checkpoint")
            axes[i].axis('off')
            continue
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        model = build_model(name, num_classes).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        
        y_true, y_pred = predict(model, test_loader, device)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        results.append({
            'model': name,
            'accuracy': acc,
            'macro_f1': f1,
            'val_acc': checkpoint.get('val_acc', None),
        })
        
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=classes, yticklabels=classes,
                    ax=axes[i], cmap='Blues', cbar=False)
        axes[i].set_title(f"{name}\nacc={acc:.3f}  F1={f1:.3f}")
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    # Hide the unused 8th subplot
    if len(ALL_MODELS) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig("results/confusion_matrices.png", dpi=120, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved results/confusion_matrices.png")
    
    # ----- Comparison bar chart -----
    df = pd.DataFrame(results)
    df = df.sort_values('accuracy', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#7f7f7f' if 'fusion' not in m else '#1f77b4' for m in df['model']]
    bars = ax.barh(df['model'], df['accuracy'], color=colors)
    ax.set_xlabel('Test Accuracy')
    ax.set_title('Model Comparison — Single-modality vs Fusion')
    ax.set_xlim(0, 1.05)
    ax.axvline(x=1/num_classes, color='red', linestyle='--', alpha=0.5,
               label=f'Random baseline ({1/num_classes:.2f})')
    
    for bar, acc in zip(bars, df['accuracy']):
        ax.text(acc + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc:.3f}', va='center')
    
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("results/model_comparison.png", dpi=120, bbox_inches='tight')
    plt.close()
    print("✓ Saved results/model_comparison.png")
    
    # ----- Save CSV summary -----
    df = pd.DataFrame(results).sort_values('accuracy', ascending=False)
    df.to_csv("results/comparison.csv", index=False)
    print("✓ Saved results/comparison.csv")
    
    # ----- Print summary -----
    print("\n" + "="*60)
    print("RESULTS SUMMARY (sorted by test accuracy)")
    print("="*60)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()