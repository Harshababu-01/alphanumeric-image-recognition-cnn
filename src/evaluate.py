import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from config.config import DIGIT_MODEL_PATH, CHAR_MODEL_PATH
from src.preprocess import load_digit_data, load_character_data

def evaluate_model(mode="character"):
    """Secure evaluation protocol scaling visually alongside the active target environment."""
    print(f"Evaluating {mode.capitalize()} Neural Framework...")
    
    if mode == "digit":
        _, (x_test, y_test) = load_digit_data()
        model_path = DIGIT_MODEL_PATH
    else:
        _, (x_test, y_test) = load_character_data()
        model_path = CHAR_MODEL_PATH
        
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Evaluation Crash: Error loading {mode} matrix -> {e}")
        return

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final Loss Metric: {loss:.4f} | Total Accuracy Metric: {accuracy:.4f}")
    
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    
    cm = confusion_matrix(y_test, y_pred)
    # Dynamically scales graphing windows based on data load density (47 class grid is huge)
    figsize = (10, 8) if mode == "digit" else (24, 20)
    fig, ax = plt.subplots(figsize=figsize)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, ax=ax, include_values=(mode == "digit"))
    
    plt.title(f"Confusion Matrix ({mode.capitalize()})")
    cm_path = os.path.join(os.path.dirname(model_path), f"confusion_matrix_{mode}.png")
    
    plt.savefig(cm_path)
    print(f"Evaluation graphic saved to {cm_path}")

if __name__ == "__main__":
    import sys
    m = sys.argv[1] if len(sys.argv) > 1 else "character"
    evaluate_model(m)
