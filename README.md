# Project (C2c): Multi-Modal Tactile Material Classification (REAL-SENSOR)
**Master’s in Robotics Engineering – University of Genoa**

## Student Information
* **Name:** Hani Bouhraoua
* **Student ID:** 8314923
* **Course Modality:** Lab-Focused

---

## 1. Project Overview
The goal of this project is to develop a **multi-modal material classification system** utilizing four distinct tactile sensing modalities: **depth, image, deformation, and shear**. By integrating these heterogeneous data streams, the system aims to achieve higher robustness and accuracy in material identification compared to single-modality approaches.

---

## 2. Concrete Tasks
To fulfill the requirements of Assignment C2c, the following steps will be executed:
1. **Prepare a unified dataset** combining all four tactile modalities.
2. **Design and implement multi-modal fusion models** for material classification.
3. **Compare different fusion strategies**, such as early fusion, late fusion, or hybrid fusion.
4. **Evaluate performance** against internal single-modality baselines.
5. **Analyze robustness and accuracy** to determine if combining modalities provides significant improvements.

---

## 3. Software Stack
The development environment is built on **Ubuntu 22.04 (Linux)** utilizing the following tools:
* **Core:** Python, ROS 2 (for sensor integration).
* **Data Processing:** NumPy, Pandas, OpenCV.
* **AI & Machine Learning:** PyTorch or TensorFlow, scikit-learn.
* **Experimentation & Visualization:** Jupyter Notebook, Matplotlib, Seaborn.

---

## 4. Research Areas
This project involves deep-dives into:
* **Multi-modal learning** and fusion strategies for heterogeneous sensor data.
* **Synchronization and preprocessing** of diverse tactile modalities.
* **Comparative analysis** with single-modality baselines.
* **Robustness and accuracy metrics** in the context of tactile material classification.

---

## 5. Deliverables
At the conclusion of the project, the repository will contain:
* A **unified multi-modal tactile dataset**.
* **Implemented fusion-based classification models**.
* A comprehensive **comparison of fusion strategies**.
* A formal **evaluation report** against single-modality baselines.

---

## 6. Workflow 
1. **Fetch:** Synchronize updates from the course main repository.
2. **Implement:** Develop fusion logic and ROS 2 nodes in the local environment.
3. **Push:** Document progress and upload code for instructor review (@giangalv).
4. **Present:** Final demonstration and report delivery in June.

## 7. Experiments and Results

We trained the seven models (4 single-modality baselines + early / late / hybrid
fusion) on two tasks. Trained models are saved in the `checkpoints_*` folders and
the plots/metrics in the matching `results_*` folders:

* **3 classes, 20 epochs** (apple, orange, kiwi) — the easy task.
* **6 classes, 20 / 30 / 40 epochs** (+ banana, nectarine, peach) — the harder
  task, trained for an increasing number of epochs.

Each `results_*` folder contains the confusion matrices, the model-comparison
bar chart, and a CSV of test accuracy and macro-F1 per model.

**What we found:**
* Fusion beats the best single modality in both tasks.
* On the easy 3-class task, **late fusion** was best (~96%).
* On the hard 6-class task, **early and hybrid fusion** were best (~84% / ~82%)
  and stayed stable across the 20/30/40-epoch runs, while **late fusion dropped
  to the weakest fusion method** (~69%).
* **Shear** was consistently the weakest modality; the **raw image** the strongest
  single one.

**Deployment note:** the models generalize well on the held-out test set, but in
real-time live prediction (`live_predict.py`) they still do **not** reach the same
high accuracy. The likely causes are that live presses differ from the recorded
trials (force, contact location, no peak-frame selection), the single specimen
per class, and the small dataset.

# Interface for Daimon tactile sensor (DM-Tac WS) Vision-based Tactile Sensor

# How to Use

## Setup
Works with Python 3.8–3.11. Make sure CUDA toolkit 12.x is installed (otherwise edit `setup.py`), then install the sensor interface:

    pip install .

Plug in the Daimon sensor before running anything that talks to it (`record.py`, `live_predict.py`). Set `DEV_SERIAL_ID` in those files to your sensor's serial.

## Pipeline at a glance
    record.py  →  dataset/  →  train.py  →  checkpoints/  →  evaluate.py / live_predict.py
                              visualize.py reads dataset/ for inspection

## 1. Record data
Collect trials with the sensor:

    python record.py

In the window: press `r` to reset the baseline (nothing touching the gel), press the object on the gel, then `SPACE` to record 30 frames. `n` = new material, `q` = quit. Data is saved to `dataset/material=<name>/trial_<NNN>/` as four `.npy` files (rawimg, depth, deformation, shear).
**Always press `r` before each trial** — without a reset, depth/deformation/shear save as zeros.

## 2. Visualize / inspect a trial
Play back a recorded trial with fixed scales and exact numbers:

    python visualize.py                 # latest trial
    python visualize.py kiwi 1          # material + trial number
    python visualize.py --compare kiwi  # peak frame of every kiwi trial
    python visualize.py --stats kiwi    # numeric summary per trial

## 3. How the data loads
`dataset_loader.py` builds the train/val/test splits **at the trial level** (no frame leakage) and, for each trial, feeds the models the single **peak-deformation frame** of all four modalities. You don't run it directly — `train.py` and `evaluate.py` import it.

## 4. Train
Train one model, or all seven (4 baselines + early/late/hybrid fusion):

    python train.py --model early_fusion
    python train.py --all --epochs 40

Best checkpoint (by validation accuracy) is saved to `checkpoints/<model>_best.pt`.

## 5. Evaluate
Score every trained model on the test set and generate plots:

    python evaluate.py

Outputs confusion matrices, a comparison bar chart, and `comparison.csv` in `results/`.

## 6. Live prediction
Real-time classification from the sensor using all trained models:

    python live_predict.py

Press `r` to reset, press an object on the gel, `SPACE` to predict (or `c` for continuous). Make sure `CLASS_NAMES` and the `checkpoints` folder match the task you trained (3-class vs 6-class).


# Baxter:
Fork and use -> https://github.com/giangalv/baxter_rosbridge_adapter, follow the README. 
