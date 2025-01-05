from autoML import *
from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path
import os


app = FastAPI(title="AutoML API")

automl = None

@app.post("/vision_classification/train")
def train_model(train_folder: str = Form(...), learning_rate: float = Form(...), epochs: int = Form(...)):
    
    global automl
    automl = AutoML()
    automl.model_fit(train_folder, task="vision_classification", lr=learning_rate, epochs=epochs)
    return {"message": "Model training completed and model saved!"}


@app.post("/vision_classification/predict")
def predict_image(image_path: str = Form(...)):
    global automl
    test_preds, labels = automl.predict(image_path)
    return {"prediction": test_preds, "labels": labels}



@app.post("/classification/train")
def train_model(train_csv_path: str = Form(...)):
    global automl
    automl = AutoML()
    automl.model_fit(train_csv_path, "classification")
    return {"message": "Model training completed!"}

@app.post("/classification/predict")
def predict_csv(test_csv_path: str = Form(...)):
    global automl
    prediction = automl.predict(test_csv_path)
    prediction = prediction.tolist()
    return {"prediction": prediction}


@app.post("/regression/train")
def train_model(train_csv_path: str = Form(...)):
    global automl
    automl = AutoML()
    automl.model_fit(train_csv_path, "regression")
    return {"message": "Model training completed!"}

@app.post("/regression/predict")
def predict_csv(test_csv_path: str = Form(...)):
    global automl
    prediction = automl.predict(test_csv_path)
    prediction = prediction.tolist()
    return {"prediction": prediction}   

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)