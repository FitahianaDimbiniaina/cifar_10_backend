from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

MODEL_PATH = "./fitahiana_finetuned.keras"
IMG_SIZE = 224
DOWNSCALE_SIZE = 128
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

model = load_model(MODEL_PATH)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = img.resize((DOWNSCALE_SIZE, DOWNSCALE_SIZE), Image.BICUBIC)
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)

    img_array = np.array(img) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)
    return img_tensor


@app.get("/")
def root():
    return {"status": "Model API running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_bytes = await file.read()
    img_tensor = preprocess_image(file_bytes)

    preds = model.predict(img_tensor)
    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    label = CLASS_NAMES[class_idx]

    # Print to backend console
    print(f"Predicted class: {label}, Confidence: {confidence*100:.2f}%")
    print(f"Softmax probabilities: {preds.tolist()}")

    return {
        "predicted_class": label,
        "confidence": round(confidence * 100, 2),
        "probabilities": preds.tolist()
    }
