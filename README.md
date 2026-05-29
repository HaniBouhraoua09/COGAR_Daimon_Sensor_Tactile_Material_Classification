# Project (C2c): Multi-Modal Tactile Material Classification (REAL-SENSOR)
**Master's in Robotics Engineering – University of Genoa**

A multi-modal material classification system for a vision-based tactile sensor
(Daimon DM-Tac WS), combining four tactile modalities — **depth, image,
deformation, and shear** — to identify materials more robustly than any single
modality alone.

## Student Information
* **Name:** Hani Bouhraoua
* **Student ID:** 8314923
* **Course Modality:** Lab-Focused

---

## 1. Project Overview
The goal of this project is to develop a **multi-modal material classification
system** utilizing four distinct tactile sensing modalities: **depth, image,
deformation, and shear**. By integrating these heterogeneous data streams, the
system aims to achieve higher robustness and accuracy in material identification
compared to single-modality approaches.

Seven models are implemented and compared: four single-modality baselines (one
per modality) and three fusion strategies — **early**, **late**, and **hybrid**
fusion.

---

## 2. Software Stack
The development environment is built on **Ubuntu 22.04 (Linux)** using:

* **Core:** Python, ROS 2 (for sensor integration).
* **Data Processing:** NumPy, Pandas, OpenCV.
* **AI & Machine Learning:** PyTorch or TensorFlow, scikit-learn.
* **Experimentation & Visualization:** Jupyter Notebook, Matplotlib, Seaborn.

---

## 3. Research Areas
This project involves deep-dives into:

* **Multi-modal learning** and fusion strategies for heterogeneous sensor data.
* **Synchronization and preprocessing** of diverse tactile modalities.
* **Comparative analysis** with single-modality baselines.
* **Robustness and accuracy metrics** in the context of tactile material
  classification.

---

## 4. Experiments and Results

We trained the seven models (4 single-modality baselines + early / late / hybrid
fusion) on two tasks. Trained models are saved in the `checkpoints_*` folders and
the plots/metrics in the matching `results_*` folders:

* **3 classes, 20 epochs** (apple, orange, kiwi) — the easy task.
* **6 classes, 20 / 30 / 40 epochs** (+ banana, nectarine, peach) — the harder
  task, trained for an increasing number of epochs.

Each `results_*` folder contains the confusion matrices, the model-comparison
bar chart, and a CSV of test accuracy and macro-F1 per model.

### Key findings
* Fusion beats the best single modality in both tasks.
* On the easy 3-class task, **late fusion** was best (~96%).
* On the hard 6-class task, **early and hybrid fusion** were best (~84% / ~82%)
  and stayed stable across the 20/30/40-epoch runs, while **late fusion dropped
  to the weakest fusion method** (~69%).
* **Shear** was consistently the weakest modality; the **raw image** the
  strongest single one.

### Deployment note
The models generalize well on the held-out test set, but in real-time live
prediction (`live_predict.py`) they still do **not** reach the same high accuracy.
The likely causes are that live presses differ from the recorded trials (force,
contact location, no peak-frame selection), the single specimen per class, and
the small dataset.

---

## 5. How to Use

### Setup
Works with Python 3.8–3.11. Make sure CUDA toolkit 12.x is installed (otherwise
edit `setup.py`), then install the sensor interface:

    pip install .

Plug in the Daimon sensor before running anything that talks to it (`record.py`,
`live_predict.py`). Set `DEV_SERIAL_ID` in those files to your sensor's serial.

### Pipeline at a glance
    record.py  →  dataset/  →  train.py  →  checkpoints/  →  evaluate.py / live_predict.py
                              visualize.py reads dataset/ for inspection

### 5.1 Record data
Collect trials with the sensor:

    python record.py

In the window: press `r` to reset the baseline (nothing touching the gel), press
the object on the gel, then `SPACE` to record 30 frames. `n` = new material,
`q` = quit. Data is saved to `dataset/material=<name>/trial_<NNN>/` as four `.npy`
files (rawimg, depth, deformation, shear).

> **Always press `r` before each trial** — without a reset, depth/deformation/shear
> save as zeros.

### 5.2 Visualize / inspect a trial
Play back a recorded trial with fixed scales and exact numbers:

    python visualize.py                 # latest trial
    python visualize.py kiwi 1          # material + trial number
    python visualize.py --compare kiwi  # peak frame of every kiwi trial
    python visualize.py --stats kiwi    # numeric summary per trial

### 5.3 How the data loads
`dataset_loader.py` builds the train/val/test splits **at the trial level** (no
frame leakage) and, for each trial, feeds the models the single
**peak-deformation frame** of all four modalities. You don't run it directly —
`train.py` and `evaluate.py` import it.

### 5.4 Train
Train one model, or all seven (4 baselines + early/late/hybrid fusion):

    python train.py --model early_fusion
    python train.py --all --epochs 40

Best checkpoint (by validation accuracy) is saved to `checkpoints/<model>_best.pt`.

### 5.5 Evaluate
Score every trained model on the test set and generate plots:

    python evaluate.py

Outputs confusion matrices, a comparison bar chart, and `comparison.csv` in
`results/`.

### 5.6 Live prediction
Real-time classification from the sensor using all trained models:

    python live_predict.py

Press `r` to reset, press an object on the gel, `SPACE` to predict (or `c` for
continuous). Make sure `CLASS_NAMES` and the `checkpoints` folder match the task
you trained (3-class vs 6-class).

---

## 6. Baxter
Fork and use → https://github.com/giangalv/baxter_rosbridge_adapter, then follow
its README.