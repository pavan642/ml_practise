# ==============================================================================
# 0. MACOS / PYTHON 3 SAFELY HANDLED CRASH PREVENTION TRICK
# ==============================================================================
# Force-loading pyarrow (if installed) or setting the environment flag prevents 
# the low-level C++ libc++abi "mutex lock failed" crash on modern Mac architectures.
import os
import sys

try:
    import pyarrow
except ImportError:
    pass

# Suppress unnecessary C++ logging noise in Python 3 terminals
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Quick Python 3 version diagnostic check
print(f"Executing under Python version: {sys.version}")
print(f"TensorFlow version loaded: {tf.__version__}")

# ==============================================================================
# 1. PARAMETERS & PYTHON 3 DATA LOADING PIPELINE
# ==============================================================================
DATA_DIR = "dataset/"  # Make sure this directory exists on your system!
BATCH_SIZE = 32
IMG_SIZE = (180, 180)

try:
    # Load the training split natively using Python 3 generators
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Load validation split
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
except ValueError as e:
    print(f"\n[Error] Could not read dataset folder: {e}")
    print("Please make sure you have subfolders inside your dataset directory.")
    sys.exit(1)

# Extract original folder class strings
class_names = train_ds.class_names
print(f"\nSuccessfully mapped Python 3 string classes: {class_names}")

# Dynamically map the negative trigger index to isolate non-flower objects
if 'not_flower' in class_names:
    NOT_FLOWER_INDEX = class_names.index('not_flower')
else:
    print("\n[Warning] No 'not_flower' directory detected. Binary gate logic will default to class index 0.")
    NOT_FLOWER_INDEX = 0

# Optimize standard Python 3 performance pipelines with pre-fetching algorithms
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==============================================================================
# 2. CONVOLUTIONAL NEURAL NETWORK ARCHITECTURE
# ==============================================================================
num_classes = len(class_names)

model = models.Sequential([
    # Standard normalization layer
    layers.Rescaling(1./255, input_shape=(180, 180, 3)),
    
    # Modern augmentation to protect small datasets
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    
    # Feature extraction convolutions
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flattening out for final classification arrays
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================================================================
# 3. TRAINING RUN
# ==============================================================================
EPOCHS = 10  # Reduced epochs for quick local debugging validation
print("\nStarting model training pass...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Export standard structural format
model.save("flower_classifier_model.keras")
print("\nModel saved successfully as 'flower_classifier_model.keras'!")

# ==============================================================================
# 4. CONDITIONAL INFERENCE PIPELINE
# ==============================================================================
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Target evaluation file path '{image_path}' not found.")
        return

    # Modern Python 3 asset pre-processing pipeline
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Convert element to array batch slice

    # Run predictions across the multi-class dimension matrix
    predictions = model.predict(img_array)
    score = predictions[0]
    
    predicted_class_idx = np.argmax(score)
    confidence = 100 * np.max(score)
    
    # Step 1 Logic Gate: Flower vs Not-a-Flower verification
    if predicted_class_idx == NOT_FLOWER_INDEX:
        print(f"\n=== Inference Result ===")
        print(f"Status: This image is NOT a flower.")
        print(f"Confidence score: {confidence:.2f}%")
    else:
        # Step 2 Logic Gate: Identify specific flower variety name
        flower_name = class_names[predicted_class_idx]
        print(f"\n=== Inference Result ===")
        print(f"Status: This image IS identified as a flower!")
        print(f"Species: {flower_name.upper()}")
        print(f"Confidence score: {confidence:.2f}%")

# To run test assertions, uncomment the line below with a valid test path:
# predict_image("test_image.jpg")
