"""
train.py - Train one model variant. Saves the best checkpoint by val accuracy.

Usage:
    python train.py --model rawimg
    python train.py --model depth
    python train.py --model deformation
    python train.py --model shear
    python train.py --model early_fusion
    python train.py --model late_fusion
    python train.py --model hybrid_fusion

Or train ALL of them with one command:
    python train.py --all
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from dataset_loader import get_dataloaders
from models import build_model


ALL_MODELS = ['rawimg', 'depth', 'deformation', 'shear',
              'early_fusion', 'late_fusion', 'hybrid_fusion']


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            pred = logits.argmax(dim=1)
            correct += (pred == batch['label']).sum().item()
            total += len(batch['label'])
    return correct / total if total > 0 else 0


def train_one_model(model_name, epochs=40, batch_size=8, lr=1e-3, seed=42):
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_loader, val_loader, test_loader, label_map = get_dataloaders(
        batch_size=batch_size, seed=seed
    )
    num_classes = len(label_map)
    
    model = build_model(model_name, num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            loss = criterion(logits, batch['label'])
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (logits.argmax(1) == batch['label']).sum().item()
            train_total += len(batch['label'])
        
        scheduler.step()
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        val_acc = evaluate(model, val_loader, device)
        
        marker = " ★" if val_acc > best_val_acc else ""
        print(f"  Epoch {epoch+1:3d}/{epochs} | "
              f"loss={train_loss:.4f} | train_acc={train_acc:.3f} | "
              f"val_acc={val_acc:.3f}{marker}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'state_dict': model.state_dict(),
                'label_map': label_map,
                'model_name': model_name,
                'val_acc': val_acc,
            }, f"checkpoints/{model_name}_best.pt")
    
    # Final test
    checkpoint = torch.load(f"checkpoints/{model_name}_best.pt")
    model.load_state_dict(checkpoint['state_dict'])
    test_acc = evaluate(model, test_loader, device)
    
    print(f"\n  ✓ {model_name}: best val={best_val_acc:.4f}, test={test_acc:.4f}")
    return {'model': model_name, 'val_acc': best_val_acc, 'test_acc': test_acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Which model to train (or use --all)")
    parser.add_argument("--all", action="store_true", help="Train all 7 models")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    if args.all:
        results = []
        for name in ALL_MODELS:
            r = train_one_model(name, args.epochs, args.batch_size, args.lr)
            results.append(r)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"{'Model':<20} {'Val Acc':>10} {'Test Acc':>10}")
        print("-"*42)
        for r in results:
            print(f"{r['model']:<20} {r['val_acc']:>10.4f} {r['test_acc']:>10.4f}")
    
    elif args.model:
        train_one_model(args.model, args.epochs, args.batch_size, args.lr)
    else:
        parser.error("Specify --model NAME or --all")