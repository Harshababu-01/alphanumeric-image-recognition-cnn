import tensorflow as tf
from tensorflow.keras import layers, models
from config.config import INPUT_SHAPE, DIGIT_CLASSES, CHAR_CLASSES, BATCH_SIZE, EPOCHS, DIGIT_MODEL_PATH, CHAR_MODEL_PATH
from src.preprocess import load_digit_data, load_character_data
import os

def build_model(num_classes):
    """Dynamic CNN Architecture compilation mapping mathematically to mapped subset classes."""
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(mode="character"):
    """Trains specified Model pipeline flawlessly separating architectures."""
    print(f"\n--- Initiating Active Training on {mode.upper()} Neural Framework ---")
    
    if mode == "digit":
        (x_train, y_train), (x_test, y_test) = load_digit_data()
        num_classes = DIGIT_CLASSES
        save_path = DIGIT_MODEL_PATH
    else:
        (x_train, y_train), (x_test, y_test) = load_character_data()
        num_classes = CHAR_CLASSES
        save_path = CHAR_MODEL_PATH
        
    print(f"Constructing CNN Matrix optimized for {num_classes} Classification Targets")
    model = build_model(num_classes)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True)
        
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
              validation_data=(x_test, y_test), callbacks=[early_stopping])
              
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Artifact Finalized and securely stored at => {save_path}")

if __name__ == "__main__":
    import sys
    # Example direct launch mechanism
    m = sys.argv[1] if len(sys.argv) > 1 else "character"
    train_model(m)
