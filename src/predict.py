import cv2
import tensorflow as tf
import numpy as np
from config.config import DIGIT_MODEL_PATH, CHAR_MODEL_PATH, CHAR_MAPPING

# Load models ONCE (fast inference)
digit_model = None
char_model = None

def load_models():
    global digit_model, char_model
    
    if digit_model is None:
        try:
            digit_model = tf.keras.models.load_model(DIGIT_MODEL_PATH)
        except Exception as e:
            print(f"Error loading digit model: {e}")
    
    if char_model is None:
        try:
            char_model = tf.keras.models.load_model(CHAR_MODEL_PATH)
        except Exception as e:
            print(f"Error loading character model: {e}")

def preprocess_image(image):
    """Advanced preprocessing: Thresholding, Bounding Box Centering, Resizing"""
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Invalid image.")

    # 1. Background normalization check (ensure white text on black background for MNIST)
    corners = [image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]]
    if np.mean(corners) > 127: 
        image = 255 - image

    # 2. Add small smoothing
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # 3. Apply thresholding (binary image)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 4. Center the digit using bounding box
    coords = cv2.findNonZero(image)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Add a small padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        cropped = image[y:y+h, x:x+w]
        
        # Make it square to preserve aspect ratio when resizing
        length = max(w, h)
        square = np.zeros((length, length), dtype=np.uint8)
        
        y_offset = (length - h) // 2
        x_offset = (length - w) // 2
        
        square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
        image = square

    # 5. Resize to 28x28
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=(0, -1))
    elif len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)

    # 6. Normalize to [0, 1]
    image = image.astype("float32")
    if image.max() > 1.0:
        image = image / 255.0

    return image

def predict(image, mode="digit"):
    load_models()
    
    try:
        image = preprocess_image(image)
    except Exception as e:
        return None, None, [], f"Image processing error: {e}"

    if mode == "digit":
        model = digit_model
    else:
        model = char_model

    if model is None:
        return None, None, [], f"{mode.capitalize()} model is not available."

    predictions = model.predict(image, verbose=0)[0]
    
    # Optional logic: If digit prediction is 3 but probability of 0 is close (diff < 10%), prefer 0
    if mode == "digit":
        prob_3 = float(predictions[3])
        prob_0 = float(predictions[0])
        
        if np.argmax(predictions) == 3 and (prob_3 - prob_0) < 0.10:
            predictions[0] += 0.20  # Boost 0 so it becomes the new max
    
    top_indices = np.argsort(predictions)[-3:][::-1]
    top_3 = []
    
    for idx in top_indices:
        # Prevent floating point anomalies after boost
        conf = min(float(predictions[idx]), 1.0)
        
        if mode == "character":
            label = CHAR_MAPPING.get(int(idx), str(idx))
        else:
            label = str(idx)
        top_3.append({"label": label, "confidence": conf})
        
    best_label = top_3[0]["label"]
    best_confidence = top_3[0]["confidence"]

    return best_label, best_confidence, top_3, None