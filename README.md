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

## 6. Repository Structure (Planned)
* `/docs`: Research notes and literature review.
* `/scripts`: ROS 2 nodes for data synchronization and acquisition.
* `/models`: Source code for Early, Late, and Hybrid fusion architectures.
* `/notebooks`: Jupyter Notebooks for data analysis and visualization.
* `/results`: Comparison plots and performance metrics.

---

## 7. Workflow 
1. **Fetch:** Synchronize updates from the course main repository.
2. **Implement:** Develop fusion logic and ROS 2 nodes in the local environment.
3. **Push:** Document progress and upload code for instructor review (@giangalv).
4. **Present:** Final demonstration and report delivery in June.
