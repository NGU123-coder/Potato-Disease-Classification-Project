import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Configuration
MODEL_PATH = "saved_models/1"
IMAGE_PATHS = {
    "Early Blight": "training/PlantVillage/Potato___Early_blight/001187a0-57ab-4329-baff-e7246a9edeb0___RS_Early.B 8178.JPG",
    "Late Blight": "training/PlantVillage/Potato___Late_blight/0051e5e8-d1c4-4a84-bf3a-a426cdad6285___RS_LB 4640.JPG",
    "Healthy": "training/PlantVillage/Potato___healthy/00fc2ee5-729f-4757-8aeb-65c3355874f2___RS_HL 1864.JPG"
}
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

print("Loading model from:", MODEL_PATH)
model = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint="serving_default")

def run_test(img_path, label):
    print(f"\n--- Testing {label} ---")
    if not os.path.exists(img_path):
        print(f"Error: Image {img_path} not found.")
        return

    img = Image.open(img_path).convert("RGB").resize((256, 256))
    img_array = np.array(img).astype(np.float32)

    # Test Case 1: Manual rescaling (what main.py does)
    img_rescaled = img_array / 255.0
    batch_rescaled = np.expand_dims(img_rescaled, axis=0)
    pred_rescaled = model(batch_rescaled)
    
    # Try to find the correct output key
    output_key = list(pred_rescaled.keys())[0]
    res_val = pred_rescaled[output_key].numpy()[0]
    
    print(f"Manual rescaling (/255): {res_val}")
    print(f"Predicted: {CLASS_NAMES[np.argmax(res_val)]} (Conf: {np.max(res_val):.4f})")

    # Test Case 2: No manual rescaling (let model do it)
    batch_raw = np.expand_dims(img_array, axis=0)
    pred_raw = model(batch_raw)
    raw_val = pred_raw[output_key].numpy()[0]
    
    print(f"No manual rescaling:      {raw_val}")
    print(f"Predicted: {CLASS_NAMES[np.argmax(raw_val)]} (Conf: {np.max(raw_val):.4f})")

for label, path in IMAGE_PATHS.items():
    run_test(path, label)
