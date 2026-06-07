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

MODEL = tf.keras.layers.TFSMLayer(
    str(MODEL_PATH),
    call_endpoint="serving_default"
)

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

    # Convert RGBA → RGB
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize((256, 256))

    image = np.array(image)

    return image
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()

    image = read_file_as_image(data)

    print("\n" + "=" * 50)
    print("Uploaded File:", file.filename)
    print("Image Shape:", image.shape)

    img_batch = np.expand_dims(image, axis=0)

    # Note: Manual rescaling (/ 255.0) removed because the model 
    # already contains a tf.keras.layers.Rescaling layer.
    img_batch = img_batch.astype(np.float32)

    prediction = MODEL(img_batch)

    prediction = prediction["output_0"].numpy()

    print("Raw Prediction Array:")
    print(prediction)

    predicted_index = np.argmax(prediction[0])

    print("Predicted Index:", predicted_index)

    print("\nClass Probabilities:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(
            f"{class_name}: {prediction[0][i] * 100:.2f}%"
        )

    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(np.max(prediction[0]))

    print("\nFinal Prediction:")
    print("Class:", predicted_class)
    print("Confidence:", confidence * 100)

    print("=" * 50 + "\n")

    return {
        "class": predicted_class,
        "confidence": round(confidence * 100, 2)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)