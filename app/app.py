import argparse
import numpy as np
import cv2
from src.predict import predict_digit
from config.config import DATA_PROCESSED_DIR
import os

def check_and_invert_background(img):
    """
    Intelligently determines if the image has a light background.
    MNIST requires a black background with white digits, so this inverts if necessary.
    """
    # Sample the 4 corners to estimate the background color
    corners = [img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]
    bg_color = np.mean(corners)
    
    if bg_color > 127: # Light background detected
        return 255 - img
    return img

def main():
    parser = argparse.ArgumentParser(description="Handwritten Digit Recognition Interface")
    parser.add_argument("--image", type=str, help="Path to a user-provided image file.")
    parser.add_argument("--sample", action="store_true", help="Predict on a random sample from the test data.")
    
    args, _ = parser.parse_known_args()
    
    if args.sample:
        sample_path = os.path.join(DATA_PROCESSED_DIR, "x_test_sample.npy")
        label_path = os.path.join(DATA_PROCESSED_DIR, "y_test_sample.npy")
        if not os.path.exists(sample_path):
            print("Processed sample data not found. Please run training or evaluate step first to generate it.")
            return
            
        x_samples = np.load(sample_path)
        y_samples = np.load(label_path)
        idx = np.random.randint(0, len(x_samples))
        
        image = x_samples[idx]
        actual_label = y_samples[idx]
        
        print(f"Randomly selected sample at index {idx}.")
        print(f"Actual Label: {actual_label}")
        
        pred, conf = predict_digit(image)
        if pred is not None:
            print(f"Predicted Digit: {pred} (Confidence: {conf:.4f})")
        
    elif args.image:
        if not os.path.exists(args.image):
            print(f"Error: Provided image '{args.image}' not found.")
            return
            
        # Read image in grayscale format
        img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        
        # Optimize preprocessing: Apply slight Gaussian Blur to reduce random noises
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Resize to expected 28x28 input cleanly
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Optimize preprocessing: ensure correct background color polarity
        img = check_and_invert_background(img)
            
        pred, conf = predict_digit(img)
        if pred is not None:
            print(f"Predicted Digit for {args.image}: {pred} (Confidence: {conf:.4f})")
        
    else:
        print("Please provide a path using --image or use --sample to predict a random sample.")
        print("Run 'python main.py app -h' for help.")

if __name__ == "__main__":
    main()
