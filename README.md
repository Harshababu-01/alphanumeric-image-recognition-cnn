# 🔤 Alphanumeric Image Recognition using CNN

## 📌 Overview

This project is a **Deep Learning-based web application** that recognizes handwritten digits and characters using a trained **Convolutional Neural Network (CNN)**.

The system provides an interactive interface where users can:

* 📤 Upload handwritten images
* ✏️ Draw characters directly on a canvas
* 🤖 Get real-time predictions with confidence percentage

It processes the input image and predicts the most probable character/digit along with how confident the model is.

---

## 🎯 Objective

To build an intelligent recognition system that:

* Accepts handwritten input (image or drawing)
* Uses deep learning to analyze patterns
* Predicts digits/characters accurately
* Displays confidence score for predictions

---

## ⚙️ How It Works

1. User uploads or draws a handwritten input
2. Image is preprocessed (resized, normalized, grayscale)
3. CNN model extracts features from the image
4. Model predicts the most likely class
5. Output is displayed with:

   * Predicted value
   * Confidence percentage (%)

---

## 🧠 Model Architecture

* Convolutional Neural Network (CNN)
* Built using:

  * TensorFlow
  * Keras

### Pipeline:

* Data Loading (MNIST Dataset)
* Image Preprocessing
* Model Training
* Model Evaluation
* Prediction Interface

---

## 📊 Dataset Used

* **MNIST Dataset**

  * 70,000 handwritten digit images (0–9)
  * Industry-standard dataset for image classification

---

## 💻 Tech Stack

* **Backend:** Python, Flask
* **Deep Learning:** TensorFlow, Keras
* **Frontend:** HTML, CSS, JavaScript
* **Libraries:** NumPy, OpenCV

---

## ✨ Key Features

* Upload handwritten images
* Draw characters directly on canvas
* Real-time prediction
* Confidence percentage output
* Modular and scalable architecture
* Clean and interactive UI

---

## 📸 Application Flow

* Upload / Draw Input
* Image Processing
* Prediction Output
* Confidence Display

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python main.py train
```

### Evaluate Model

```bash
python main.py evaluate
```

### Run Application

```bash
python main.py app --sample
```

OR for custom input:

```bash
python main.py app --image ./path-to-image.png
```

---

## 🧪 Testing

```bash
pytest
```

---



---
