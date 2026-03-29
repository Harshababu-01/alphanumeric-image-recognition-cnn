# Handwritten Digit Recognition with CNN

A professional, modular machine learning project that implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to predictably recognize handwritten digits from the MNIST dataset.

## Architecture & Modular Structure

```text
├── app/                  # Application interface (CLI functionality)
│   └── app.py
├── config/               # Configuration and hyperparameter constants
│   └── config.py
├── data/                 # Data folder
│   ├── raw/              # Raw data downloads (git ignored)
│   └── processed/        # Processed / ready datasets (git ignored)
├── models/               # Saved trained artifacts (git ignored)
├── src/                  # Source code for the ML pipeline
│   ├── preprocess.py     # Data fetching and preprocessing
│   ├── train.py          # Network topology and model training
│   ├── evaluate.py       # Metrics evaluation module
│   └── predict.py        # Independent inference script
├── tests/                # Unit testing suite
│   └── test_model.py
├── utils/                # Utilities and generic helper scripts
│   └── helper.py
├── main.py               # Main orchestrator / entrypoint
├── requirements.txt      # Project library dependencies
└── .gitignore            # Git exclusion settings
```

## Setup & Installation

It is best to set up an isolated Python environment using \`venv\` or \`conda\`.

1. Create and activate a Virtual Environment:
   ```bash
   python -m venv venv
   # Depending on OS:
   source venv/bin/activate      # Mac/Linux
   venv\Scripts\activate         # Windows
   ```

2. Install Project Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Pipeline Commands

The orchestrator script, `main.py`, exposes the major actions:

### 1. Train the Network
Downloads the MNIST dataset locally (into ~/.keras caching directory conceptually mapped to raw/ format), extracts preprocessing samples, builds the CNN architecture, and finally evaluates logic to `models/mnist_cnn.keras`:
```bash
python main.py train
```

### 2. Model Evaluation
Will invoke the pre-trained module and summarize loss & accuracy metrics across the test dataset:
```bash
python main.py evaluate
```

### 3. Application Interface
Leverage your model to derive digit predictions! Test out a random sample or specify a custom grayscale image layout:
```bash
# Random Test Inference
python main.py app --sample

# Custom Hand-Drawn File Input
python main.py app --image ./path-to-image.png
```

## Testing Context
Tests can be triggered globally employing `pytest` from the root layer:
```bash
pytest
```
