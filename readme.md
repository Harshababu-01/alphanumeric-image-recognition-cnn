# Alphanumeric Image Recognition System (CNN)

A deep learning-based web application that recognizes **alphanumeric characters (digits + letters)** from both uploaded images and user-drawn input using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

---

## 🎯 Overview

This project provides an intelligent system capable of:

* ✍️ Recognizing characters drawn on a canvas
* 📤 Predicting characters from uploaded images
* 🔢 Supporting digits (0–9) and extending towards full alphanumeric recognition
* 🤖 Delivering predictions with confidence scores

---

## 🧠 Model & Dataset

* CNN (Convolutional Neural Network)
* Built using TensorFlow & Keras

### Dataset Used:

* MNIST Dataset (Digits 0–9)

> ⚠️ Note: Character (A–Z) recognition is designed as an extension and can be integrated with additional datasets.

---

## ✨ Features

* Upload image for prediction
* Draw characters on canvas
* Real-time prediction
* Confidence score output
* Modular ML pipeline
* Clean Flask-based web interface

---

## 🏗️ Architecture & Modular Structure

```text
├── app/                  
│   └── app.py
├── config/               
│   └── config.py
├── data/                 
│   ├── raw/              
│   └── processed/        
├── models/               
├── src/                  
│   ├── preprocess.py     
│   ├── train.py          
│   ├── evaluate.py       
│   └── predict.py        
├── tests/                
│   └── test_model.py
├── utils/                
│   └── helper.py
├── static/               
├── templates/            
├── uploads/              
├── main.py               
├── app.py                
├── requirements.txt      
└── .gitignore            
```

---

## 💻 Tech Stack

* Python
* TensorFlow / Keras
* Flask
* HTML, CSS, JavaScript
* NumPy

---

## ⚙️ Setup & Installation

### 1. Create & Activate Environment

```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

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

### Custom Image

```bash
python main.py app --image ./path-to-image.png
```

---

## 📸 Application Flow

* Upload / Draw Input
* Image Preprocessing
* CNN Prediction
* Result with Confidence Score

---

## 🧪 Testing

```bash
pytest
```


