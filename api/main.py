import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Robust path handling for model loading
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "saved_models" / "1"

# Load model using low-level API compatible with TF 2.15
model_loaded = tf.saved_model.load(str(MODEL_PATH))
MODEL_SERVE = model_loaded.signatures["serving_default"]

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Production CORS settings
# Set FRONTEND_URL environment variable in Render (e.g., https://your-app.vercel.app)
origins = [
    "http://localhost",
    "http://localhost:3000",
    os.getenv("FRONTEND_URL", "*")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
def ping():
    return "Hello I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    image = read_file_as_image(data)

    img_batch = np.expand_dims(image, axis=0)
    img_batch = tf.constant(img_batch.astype(np.float32))

    # Inference using signature
    predictions = MODEL_SERVE(img_batch)
    
    # In SavedModel signatures, the output key is typically 'output_0' or 'Identity'
    # Based on our previous diagnostic, it was 'output_0'
    output_key = list(predictions.keys())[0]
    prediction = predictions[output_key].numpy()

    predicted_index = np.argmax(prediction[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(np.max(prediction[0]))

    return {
        "class": predicted_class,
        "confidence": round(confidence * 100, 2)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)