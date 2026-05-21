"""
live_predict.py - Real-time tactile material classification.

Plug in the Daimon sensor, press an object on the gel, hit SPACE,
and see the prediction from all 7 trained models simultaneously.

Controls:
    SPACE  - Run prediction on current sensor reading
    r      - Reset sensor baseline (do this BEFORE pressing an object!)
    c      - Continuous mode (predict every ~0.5 seconds)
    s      - Stop continuous mode
    q      - Quit
"""

import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dmrobotics import Sensor, put_arrows_on_image
from models import build_model


# ============================================================
# CONFIGURATION
# ============================================================
DEV_SERIAL_ID = "S2508080077"
CHECKPOINT_DIR = "checkpoints"
ALL_MODELS = ['rawimg', 'depth', 'deformation', 'shear',
              'early_fusion', 'late_fusion', 'hybrid_fusion']
CLASS_NAMES = ['apple', 'banana', 'kiwi', 'nectarine', 'orange', 'peach']  # ← must match training label_map order!

# Visual settings
PANEL_BG = (30, 30, 30)
TEXT_GOOD = (0, 255, 100)    # green
TEXT_OK   = (0, 200, 255)    # blue
TEXT_BAD  = (0, 100, 255)    # orange
TEXT_GREY = (180, 180, 180)
TEXT_GOLD = (0, 215, 255)    # gold (best model)


# ============================================================
# LOAD ALL MODELS ONCE
# ============================================================
def load_all_models(device):
    """Load all 7 trained models from checkpoints/."""
    models = {}
    print("Loading models...")
    for name in ALL_MODELS:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{name}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [skip] {name}: checkpoint not found")
            continue
        
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(name, num_classes=len(CLASS_NAMES)).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        models[name] = model
        
        val_acc = checkpoint.get('val_acc', '?')
        if isinstance(val_acc, float):
            val_acc = f"{val_acc:.3f}"
        print(f"  ✓ {name}: loaded (val_acc={val_acc})")
    
    if not models:
        sys.exit("No models loaded! Run train.py first.")
    
    print(f"\nLoaded {len(models)} models. Ready for live prediction.\n")
    return models


# ============================================================
# PREPARE INPUT FOR MODELS
# ============================================================
def prepare_input(raw_img, depth, deformation, shear, device):
    """Convert raw sensor data into a model-ready batch dict."""
    # Normalize to model input format (same as dataset_loader.py)
    rawimg_t = torch.from_numpy(raw_img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    depth_t  = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    defo_t   = torch.from_numpy(deformation.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    shear_t  = torch.from_numpy(shear.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    
    batch = {
        'rawimg':      rawimg_t.to(device),
        'depth':       depth_t.to(device),
        'deformation': defo_t.to(device),
        'shear':       shear_t.to(device),
    }
    return batch


# ============================================================
# RUN ALL MODELS, GET PREDICTIONS
# ============================================================
def predict_all(models, batch):
    """Run all models on the same input. Return dict: {model_name: (pred_class, confidence, all_probs)}."""
    results = {}
    with torch.no_grad():
        for name, model in models.items():
            logits = model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # (num_classes,)
            pred_class = int(probs.argmax())
            confidence = float(probs[pred_class])
            results[name] = {
                'class': pred_class,
                'class_name': CLASS_NAMES[pred_class],
                'confidence': confidence,
                'probs': probs,
            }
    return results


# ============================================================
# DRAW THE PREDICTION PANEL
# ============================================================
def draw_predictions_panel(results, panel_w=600, panel_h=480):
    """Create a visual panel showing all model predictions."""
    panel = np.full((panel_h, panel_w, 3), PANEL_BG, dtype=np.uint8)
    
    # Title
    cv2.putText(panel, "LIVE PREDICTIONS", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.line(panel, (20, 50), (panel_w - 20, 50), (100, 100, 100), 1)
    
    # Sort: fusion models first (highlighted), then baselines
    fusion_order = ['late_fusion', 'hybrid_fusion', 'early_fusion']
    baseline_order = ['rawimg', 'depth', 'deformation', 'shear']
    display_order = [m for m in fusion_order if m in results] + \
                    [m for m in baseline_order if m in results]
    
    # Find the best fusion result (for "winning" highlight)
    best_model = max(results.keys(), key=lambda k: results[k]['confidence']) \
                 if results else None
    
    y = 90
    for i, name in enumerate(display_order):
        r = results[name]
        cls = r['class_name']
        conf = r['confidence']
        
        # Color by confidence
        if conf > 0.85:
            color = TEXT_GOOD
        elif conf > 0.60:
            color = TEXT_OK
        else:
            color = TEXT_BAD
        
        # Highlight late_fusion (best model) with gold
        is_best = (name == 'late_fusion')
        label_color = TEXT_GOLD if is_best else TEXT_GREY
        
        # Separator before baselines
        if name == baseline_order[0] if baseline_order else None:
            cv2.line(panel, (20, y - 12), (panel_w - 20, y - 12), (80, 80, 80), 1)
            cv2.putText(panel, "Single-modality baselines:", (20, y - 18 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            y += 22
        
        # Model name
        prefix = "★ " if is_best else "  "
        cv2.putText(panel, f"{prefix}{name}:", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 1)
        
        # Predicted class and confidence
        result_text = f"{cls.upper()}  ({conf*100:.0f}%)"
        cv2.putText(panel, result_text, (260, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2 if is_best else 1)
        
        # Mini confidence bar
        bar_x = 470
        bar_w = 110
        bar_h = 12
        bar_y = y - 10
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (60, 60, 60), -1)
        fill_w = int(bar_w * conf)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      color, -1)
        
        y += 35
    
    # Footer with instructions
    footer_y = panel_h - 40
    cv2.line(panel, (20, footer_y - 10), (panel_w - 20, footer_y - 10),
             (100, 100, 100), 1)
    cv2.putText(panel, "SPACE=predict  r=reset  c=continuous  q=quit",
                (20, footer_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (180, 180, 180), 1)
    
    return panel


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load all trained models
    models = load_all_models(device)
    
    # Connect to sensor
    print(f"Connecting to Daimon sensor (serial: {DEV_SERIAL_ID})...")
    sensor = Sensor(DEV_SERIAL_ID)
    print("Sensor connected.\n")
    
    # Setup
    black_img = np.zeros_like(sensor.getRawImage())
    black_img = np.stack([black_img] * 3, axis=-1)
    
    print("=" * 60)
    print("LIVE TACTILE MATERIAL CLASSIFICATION")
    print("=" * 60)
    print("Controls:")
    print("  SPACE - Predict from current sensor reading")
    print("  r     - Reset sensor baseline (do this BEFORE pressing!)")
    print("  c     - Toggle continuous prediction mode")
    print("  q     - Quit")
    print("=" * 60)
    
    # Initial empty panel
    last_results = None
    last_panel = np.full((480, 600, 3), PANEL_BG, dtype=np.uint8)
    cv2.putText(last_panel, "Press 'r' to reset, then press an object",
                (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(last_panel, "on the gel and hit SPACE.",
                (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    continuous_mode = False
    last_prediction_time = 0
    
    try:
        while True:
            # Live sensor read
            img = sensor.getRawImage()
            depth = sensor.getDepth()
            deformation = sensor.getDeformation2D()
            shear = sensor.getShear()
            
            # Live preview windows (like record.py)
            depth_img = cv2.applyColorMap(
                (depth * 0.25 * 255.0).astype('uint8'), cv2.COLORMAP_HOT
            )
            cv2.imshow('img', img)
            cv2.imshow('depth', depth_img)
            cv2.imshow('deformation', put_arrows_on_image(black_img, deformation * 20))
            cv2.imshow('shear', put_arrows_on_image(black_img, shear * 20))
            cv2.imshow('predictions', last_panel)
            
            # Continuous prediction mode
            if continuous_mode and (time.time() - last_prediction_time > 0.5):
                batch = prepare_input(img, depth, deformation, shear, device)
                last_results = predict_all(models, batch)
                last_panel = draw_predictions_panel(last_results)
                last_prediction_time = time.time()
            
            # Handle keys
            k = cv2.waitKey(15) & 0xFF
            
            if k == ord('q'):
                print("\nQuitting...")
                break
            
            elif k == ord('r'):
                sensor.reset()
                print("[RESET] Sensor baseline zeroed. Now press an object and hit SPACE.")
            
            elif k == ord('c'):
                continuous_mode = not continuous_mode
                status = "ON" if continuous_mode else "OFF"
                print(f"[CONTINUOUS MODE] {status}")
            
            elif k == ord(' '):  # SPACE = predict once
                print("\n[PREDICT] Running all models...")
                batch = prepare_input(img, depth, deformation, shear, device)
                t0 = time.time()
                last_results = predict_all(models, batch)
                elapsed = time.time() - t0
                last_panel = draw_predictions_panel(last_results)
                
                # Print to terminal
                print(f"  Inference time: {elapsed*1000:.1f} ms")
                print(f"  {'Model':<20} {'Prediction':<10} {'Confidence':>10}")
                print("  " + "-" * 42)
                for name in ['late_fusion', 'hybrid_fusion', 'early_fusion',
                             'rawimg', 'depth', 'deformation', 'shear']:
                    if name in last_results:
                        r = last_results[name]
                        marker = " ★" if name == 'late_fusion' else "  "
                        print(f"  {marker}{name:<18} {r['class_name']:<10} "
                              f"{r['confidence']*100:>9.1f}%")
                print()
    
    finally:
        sensor.disconnect()
        cv2.destroyAllWindows()
        print("Sensor disconnected. Bye!")


if __name__ == "__main__":
    main()