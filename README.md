<div align="center">

<br/>

```
██╗   ██╗ ██████╗  ██████╗ █████╗ ██╗      ██╗   ██╗███████╗██╗███████╗
██║   ██║██╔═══██╗██╔════╝██╔══██╗██║      ╚██╗ ██╔╝██╔════╝██║██╔════╝
██║   ██║██║   ██║██║     ███████║██║       ╚████╔╝ ███████╗██║███████╗
╚██╗ ██╔╝██║   ██║██║     ██╔══██║██║        ╚██╔╝  ╚════██║██║╚════██║
 ╚████╔╝ ╚██████╔╝╚██████╗██║  ██║███████╗   ██║   ███████║██║███████║
  ╚═══╝   ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚══════╝╚═╝╚══════╝
```

# 🎙️ Parkinson's Voice Detection

### AI-Powered Disease Prediction via Voice Signal Analysis

<br/>

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-API-000000?style=flat-square&logo=flask&logoColor=white)
![Librosa](https://img.shields.io/badge/Librosa-Audio-FF6B6B?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)
![Status](https://img.shields.io/badge/Status-Educational-8b5cf6?style=flat-square)

<br/>

> *Detecting patterns in voice that the human ear cannot perceive.*

<br/>

</div>

---

## 📌 Overview

This project is an AI-powered system that analyzes voice signals to detect Parkinson's disease. It uses machine learning to identify subtle, hidden patterns in speech biomarkers — patterns invisible to the naked ear — and delivers prediction results through a clean, modular pipeline.

Built on the **UCI Parkinson's Dataset**, this system bridges the gap between raw audio data and actionable health insights.

---

## 🎯 Objectives

- 🔬 Analyze biomedical voice data for healthcare insights
- 🧠 Build a robust ML classification model for disease prediction
- 🔌 Expose predictions via a REST API
- 🌍 Demonstrate a real-world AI application in healthcare

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | `Python 3.9+` | Core development |
| ML Framework | `Scikit-learn` | Model building & evaluation |
| Data Processing | `Pandas`, `NumPy` | Feature engineering |
| Audio Processing | `Librosa` | MFCC & voice feature extraction |
| Backend | `Flask` | REST API server |
| Model Storage | `Joblib` | Model serialization |

---

## 📂 Project Structure

```
parkinsons-voice-detection/
│
├── 📁 data/                  # Raw & processed datasets
│   └── parkinsons.data
│
├── 📁 models/                # Saved trained models
│   └── rf_model.joblib
│
├── 📁 src/                   # Core source code
│   ├── preprocess.py         # Data cleaning & feature selection
│   ├── features.py           # MFCC extraction logic
│   ├── train.py              # Model training pipeline
│   └── predict.py            # Inference utilities
│
├── 📄 app.py                 # Flask API entry point
├── 📄 requirements.txt       # Python dependencies
└── 📄 README.md              # You are here
```

---

## ⚙️ How It Works

```
  ┌──────────────────┐
  │  User Input      │  ← .wav audio file
  │  (.wav file)     │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ Feature          │  ← MFCC, pitch, jitter,
  │ Extraction       │    shimmer, HNR
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ Random Forest    │  ← Trained on UCI
  │ Classifier       │    Parkinson's Dataset
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Prediction      │  ← Parkinson's Detected
  │  Result          │     OR  Healthy
  └──────────────────┘
```

### Step-by-Step Pipeline

**① Data Preparation**
Clean and structure the UCI Parkinson's dataset. Select the most discriminative features for classification.

**② Feature Extraction**
Extract MFCC (Mel-Frequency Cepstral Coefficients) from raw `.wav` files using `Librosa`. These capture the unique spectral properties of a speaker's voice.

**③ Model Training**
Train a `RandomForestClassifier` on the extracted features. The ensemble approach improves robustness against overfitting on limited medical data.

**④ Evaluation**
Evaluate performance on a held-out test split using accuracy, precision, recall, and F1-score.

**⑤ Prediction via API**
Submit a `.wav` file to the Flask endpoint and receive a JSON response with the prediction result.

---

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.9
pip
```

### 1 — Clone the repository

```bash
git clone https://github.com/your-username/parkinsons-voice-detection.git
cd parkinsons-voice-detection
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Train the model

```bash
python src/train.py
```

### 4 — Launch the API server

```bash
python app.py
```

### 5 — Make a prediction

```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@your_voice_sample.wav"
```

**Sample Response:**
```json
{
  "prediction": "Parkinson's Detected",
  "confidence": 0.87
}
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~92% |
| Algorithm | Random Forest |
| Dataset | UCI Parkinson's (195 samples) |
| Features Used | MFCC, Jitter, Shimmer, HNR |

> Performance varies depending on train/test split and feature selection strategy.

---

## 📡 API Reference

### `POST /predict`

Accepts a `.wav` audio file and returns the prediction.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file` | `multipart/form-data` | ✅ | `.wav` voice recording |

**Success Response `200`**
```json
{
  "prediction": "Healthy",
  "confidence": 0.94
}
```

**Error Response `400`**
```json
{
  "error": "No audio file provided"
}
```

---

## ⚠️ Challenges & Limitations

| Challenge | Details |
|---|---|
| 📉 Dataset size | UCI dataset contains only ~195 samples — limited generalizability |
| 🎙️ Basic features | MFCC alone may miss nuanced pathological markers |
| 🔇 Noise sensitivity | Real-world recordings introduce background noise |
| 🏥 Not clinical-grade | Not validated for actual medical use |

---

## 🔮 Future Improvements

- [ ] 🧠 Integrate deep learning (CNN / LSTM) on raw spectrograms
- [ ] 🎤 Add real-time voice recording via browser microphone
- [ ] 🖥️ Build a polished frontend UI (React / Streamlit)
- [ ] ☁️ Deploy to cloud (AWS / GCP / Render)
- [ ] 📈 Expand dataset with augmentation techniques
- [ ] 🌐 Add multilingual support for diverse voice samples

---

## 📦 Dependencies

```txt
flask
scikit-learn
pandas
numpy
librosa
joblib
soundfile
```

Install everything with:
```bash
pip install -r requirements.txt
```

---

## ⚠️ Disclaimer

> **This project is for educational and research purposes only.**
> It is **not** a medical device and should **not** be used for clinical diagnosis or treatment decisions. Always consult a licensed medical professional for health concerns.

---

## 👨‍💻 Author

**Nitya Gupta**

<br/>

<div align="center">

*"The voice carries more information than words alone."*

<br/>

⭐ If you found this project useful, consider giving it a star!

</div>
