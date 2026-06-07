import os
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image
import tensorflow as tf
import uvicorn

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ==========================
# MODEL LOADING
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "saved_models" / "1"

print("Loading model from:", MODEL_PATH)

model_loaded = tf.saved_model.load(str(MODEL_PATH))
MODEL_SERVE = model_loaded.signatures["serving_default"]

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# ==========================
# CORS CONFIGURATION
# ==========================

FRONTEND_URL = os.getenv("FRONTEND_URL")

origins = [
    "http://localhost",
    "http://localhost:3000",
]

if FRONTEND_URL:
    origins.append(FRONTEND_URL)

print("FRONTEND_URL =", FRONTEND_URL)
print("ALLOWED ORIGINS =", origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# HEALTH CHECK
# ==========================

@app.get("/")
def root():
    return {"message": "Potato Disease API Running"}

@app.get("/ping")
def ping():
    return {"message": "Hello I am alive"}

@app.get("/debug-cors")
def debug_cors():
    return {
        "frontend_url": FRONTEND_URL,
        "allowed_origins": origins
    }

# ==========================
# IMAGE PROCESSING
# ==========================

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))

    image = image.convert("RGB")
    image = image.resize((256, 256))

    image = np.array(image)

    return image

# ==========================
# PREDICTION ENDPOINT
# ==========================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        print("\n" + "=" * 50)
        print("Prediction Request Received")
        print("File Name:", file.filename)

        data = await file.read()

        image = read_file_as_image(data)

        print("Image Shape:", image.shape)

        img_batch = np.expand_dims(image, axis=0)

        img_batch = tf.constant(
            img_batch.astype(np.float32)
        )

        print("Running inference...")

        predictions = MODEL_SERVE(img_batch)

        output_key = list(predictions.keys())[0]

        prediction = predictions[output_key].numpy()

        print("Raw Prediction:", prediction)

        predicted_index = int(np.argmax(prediction[0]))

        predicted_class = CLASS_NAMES[predicted_index]

        confidence = float(
            np.max(prediction[0])
        )

        print("Predicted Class:", predicted_class)
        print("Confidence:", confidence)

        print("=" * 50)

        return {
            "class": predicted_class,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {
            "error": str(e)
        }

# ==========================
# LOCAL RUN
# ==========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )