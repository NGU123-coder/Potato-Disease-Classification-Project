from fastapi import FastAPI, File, UploadFile
import uvicorn
import requests
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
def ping():
    return "Hello I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))

    # Convert RGBA to RGB
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize((256, 256))

    # Convert to numpy array
    image = np.array(image)

    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        data = await file.read()

        image = read_file_as_image(data)

        print("Image Shape:", image.shape)

        # Create batch dimension
        img_batch = np.expand_dims(image, axis=0)

        # Normalize image
        img_batch = img_batch.astype(np.float32) / 255.0

        # Convert numpy array to JSON serializable list
        json_data = {
            "instances": img_batch.tolist()
        }

        # Send request to TensorFlow Serving
        response = requests.post(endpoint, json=json_data)

        print("Status Code:", response.status_code)
        print("Response:", response.text)

        # Get prediction
        prediction = np.array(response.json()["predictions"])

        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]))

        return {
            "class": predicted_class,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)