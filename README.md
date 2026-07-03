# CircuitSense

**AI-Powered Handwritten Circuit Symbol Recognition using TensorFlow, MobileNetV2, and Streamlit**

CircuitSense is a deep learning application that recognizes handwritten electronic circuit symbols from images. The project uses **MobileNetV2 Transfer Learning** to classify **15 different circuit components** and provides confidence scores, top-3 predictions, and basic engineering information through an interactive Streamlit interface.



##  Features

-  Classifies **15 handwritten electronic circuit symbols**
-  MobileNetV2 Transfer Learning
-  Validation Accuracy: **~88%**
-  Top-3 Prediction Probabilities
-  Confidence Score
-  Interactive Streamlit Web Application
-  Component Function, Unit, and Applications
-  Confusion Matrix and Classification Report


##  Supported Circuit Symbols

- Ammeter
- AC Source
- Battery
- Capacitor
- Current Source
- DC Voltage Source 1
- DC Voltage Source 2
- Dependent Current Source
- Dependent Voltage Source
- Diode
- Ground Symbol 1
- Ground Symbol 2
- Inductor
- Resistor
- Voltmeter

---

## Tech Stack

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Matplotlib
- Streamlit

---


## 📈 Training Pipeline

1. Dataset Exploration
2. Data Cleaning
3. Data Augmentation
4. TensorFlow Data Pipeline
5. MobileNetV2 Transfer Learning
6. Model Evaluation
7. Streamlit Deployment


##  How It Works

1. Upload a handwritten circuit symbol.
2. The image is resized to **224 × 224**.
3. The trained MobileNetV2 model predicts the symbol.
4. The application displays:
   - Predicted symbol
   - Confidence score
   - Top-3 predictions
   - Engineering information

---

## Evaluation

The model was evaluated using:

- Validation Accuracy
- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1-Score

Overall Validation Accuracy:

**≈ 88%**

---

## Challenges Faced

- Cleaning corrupted dataset files (`Zone.Identifier`)
- Transfer learning with MobileNetV2
