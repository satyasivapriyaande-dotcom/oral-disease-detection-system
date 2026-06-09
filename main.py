from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import csv
import os
import json


app = FastAPI(title="AI Oral Cancer Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


print("Loading models...")

cancer_model = tf.keras.models.load_model("cancer_model.h5")
disease_model = tf.keras.models.load_model("disease_model.h5")
cancer_cnn_model = tf.keras.models.load_model("cnn_cancer_model.h5")
disease_cnn_model = tf.keras.models.load_model("cnn_disease_model.h5")

print("Models loaded successfully")


with open("disease_classes.json") as f:
    class_indices = json.load(f)

DISEASE_CLASSES = {v: k for k, v in class_indices.items()}


SUBCLASS_MAP = {
    "Leukoplakia": "White_Lesion",
    "Oral_Lichen": "White_Lesion",
    "Gingivitis": "Inflammatory_Lesion",
    "Mouth_Ulcer": "Inflammatory_Lesion",
    "Oral_Thrush": "Inflammatory_Lesion",
    "Sub_Mucosal_Fibrosis": "Inflammatory_Lesion",
    "Dental_Caries": "Dental_Issue",
    "Calculus": "Dental_Issue",
    "Tooth_discoloration": "Dental_Issue",
    "Hypodontia": "Developmental_Issue",
    "Healthy": "Healthy"
}


USERS_FILE = "users.csv"


@app.post("/register")
def register(data: dict = Body(...)):
    with open(USERS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([data["username"], data["password"]])
    return {"success": True}


@app.post("/login")
def login(data: dict = Body(...)):
    if not os.path.exists(USERS_FILE):
        return {"success": False}

    with open(USERS_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == data["username"] and row[1] == data["password"]:
                return {"success": True}

    return {"success": False}


@app.post("/questionnaire")
def questionnaire(data: dict = Body(...)):
    score = sum(1 for v in data.values() if v.lower() == "yes")
    return {"submitted": True, "risk_score": score}


def predict_densenet(arr):

    THRESHOLD = 0.75
    cancer_prob = float(cancer_model.predict(arr)[0][0])

    # Cancer
    if cancer_prob < (1 - THRESHOLD):
        return {
            "prediction": "Cancer",
            "disease": "Oral Cancer",
            "sub_class": "Cancer",
            "confidence": round((1 - cancer_prob) * 100, 2)
        }

    # Non-Cancer
    disease_pred = disease_model.predict(arr)[0]
    idx = int(np.argmax(disease_pred))

    disease = DISEASE_CLASSES.get(idx, "Unknown")
    sub_class = SUBCLASS_MAP.get(disease, "Other")
    confidence = round(float(disease_pred[idx]) * 100, 2)

    return {
        "prediction": "Non-Cancer",
        "disease": disease,
        "sub_class": sub_class,
        "confidence": confidence
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    
    img_cnn = img.resize((224, 224))
    arr_cnn = np.array(img_cnn) / 255.0
    arr_cnn = np.expand_dims(arr_cnn, axis=0)

    
    img_dense = img.resize((160, 160))
    arr_dense = np.array(img_dense) / 255.0
    arr_dense = np.expand_dims(arr_dense, axis=0)

   
    cnn_prob = float(cancer_cnn_model.predict(arr_cnn)[0][0])

    if cnn_prob < 0.5:
        cnn_result = {
            "prediction": "Cancer",
            "disease": "Oral Cancer",
            "sub_class": "Cancer",
            "confidence": round((1 - cnn_prob) * 100, 2)
        }
    else:
        disease_pred = disease_cnn_model.predict(arr_cnn)[0]
        idx = int(np.argmax(disease_pred))

        disease = DISEASE_CLASSES.get(idx, "Unknown")
        sub_class = SUBCLASS_MAP.get(disease, "Other")
        confidence = round(float(disease_pred[idx]) * 100, 2)

        cnn_result = {
            "prediction": "Non-Cancer",
            "disease": disease,
            "sub_class": sub_class,
            "confidence": confidence
        }

    
    dense_result = predict_densenet(arr_dense)

    
    final_result = {
        "prediction": dense_result["prediction"],
        "disease": dense_result["disease"],
        "sub_class": dense_result["sub_class"],
        "confidence": dense_result["confidence"]
    }

    
    return {
        "cnn": cnn_result,
        "densenet": dense_result,
        "final": final_result
    }