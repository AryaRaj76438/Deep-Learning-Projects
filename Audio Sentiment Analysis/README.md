# 🎧 Speech Emotion Recognition using Deep Learning (Conv1D)

This repository implements a **Speech Emotion Recognition (SER)** system using handcrafted audio features and a **1D Convolutional Neural Network (Conv1D)**.  
The model is trained and evaluated on the **TESS (Toronto Emotional Speech Set)** dataset and achieves **~98% accuracy** on the test set.

---

## 📌 Project Overview

Speech Emotion Recognition aims to identify human emotions from speech signals.  
In this project:

- Audio signals are preprocessed and augmented
- Acoustic features such as **MFCCs, Mel Spectrograms, Chroma, ZCR, and RMS** are extracted
- A **Conv1D-based deep learning model** is trained for multi-class emotion classification

---

## 🗂️ Datasets Used

The code supports multiple standard SER datasets:

- **TESS** – Toronto Emotional Speech Set ✅ *(used for training & evaluation)*
- **RAVDESS**
- **SAVEE**
- **CREMA-D**

> ⚠️ Current results are reported on **TESS dataset only**

---

## 🎭 Emotion Classes (14)

- OAF_angry  
- OAF_disgust  
- OAF_fear  
- OAF_happy  
- OAF_neutral  
- OAF_sad  
- OAF_pleasant_surprise  
- YAF_angry  
- YAF_disgust  
- YAF_fear  
- YAF_happy  
- YAF_neutral  
- YAF_sad  
- YAF_pleasant_surprised  

(OAF = Older Adult Female, YAF = Younger Adult Female)

---

## 🔊 Audio Preprocessing & Augmentation

Each audio sample is:
- Trimmed to **2.5 seconds**
- Offset by **0.6 seconds**

### Data Augmentation Techniques

- Additive Noise
- Time Stretching
- Pitch Shifting

This improves robustness and increases effective dataset size.

---

## 🎼 Feature Extraction

For each audio signal, the following features are extracted and averaged over time:

- Zero Crossing Rate (ZCR)
- Chroma STFT
- MFCCs
- RMS Energy
- Mel Spectrogram

All features are concatenated into a single feature vector and standardized using **StandardScaler**.

---

## 🧠 Model Architecture (Conv1D)
Input → Conv1D(256) → MaxPooling
→ Conv1D(256) → MaxPooling
→ Conv1D(128) → MaxPooling
→ Dropout
→ Conv1D(64) → MaxPooling
→ Flatten
→ Dense(32, ReLU)
→ Dropout
→ Dense(14, Softmax)


- Optimizer: **Adam**
- Loss Function: **Categorical Crossentropy**
- Early Stopping enabled

---

## 📊 Results

### 🔹 Overall Performance

- **Test Accuracy:** ~97.98%
- **Macro Avg F1-score:** ~0.98
- **Weighted Avg F1-score:** ~0.98

The confusion matrix shows strong diagonal dominance, indicating high class separability.

---

## ⚠️ Important Note on Evaluation

Data augmentation is applied **before** the train-test split, which may introduce **data leakage** and optimistic results.

For more robust evaluation:
- Split audio files before augmentation
- Apply augmentation only on the training set
- Perform speaker-independent evaluation

This limitation is acknowledged for academic transparency.

---

## 🛠️ Tech Stack

- Python
- Librosa
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib

---
