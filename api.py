from typing import Annotated

from autoML import *
from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path
import os
from zipfile import ZipFile
import shutil
import uuid


app = FastAPI(title="AutoML API")

automl = None


@app.post("/vision_classification/train")
async def train_model(train_folder: UploadFile = File("select zip file that include your image data"), learning_rate: float = Form(...), epochs: int = Form(...)):
    dataTemp = "./dataTemp"
    os.makedirs(dataTemp, exist_ok=True)
    
    ziptemPath = "./file.zip"
    
    #save file to disk from ram.
    with open(ziptemPath, "wb") as buffer:
        shutil.copyfileobj(train_folder.file, buffer)
        
    
    with ZipFile(ziptemPath,"r") as zObject:
        zObject.extractall(dataTemp)
    
    global automl
    automl = AutoML()
    automl.model_fit("dataTemp/images", task="vision_classification", lr=learning_rate, epochs=epochs)
    shutil.rmtree("./dataTemp")
    os.remove("./file.zip")
    return {"message": "Model training completed!"}




@app.post("/vision_classification/predict")
async def predict_image(image_path: UploadFile = File("upload your image file")):
    local_image_path = f"{image_path.filename}"
    with open(local_image_path, "wb") as buffer:
        shutil.copyfileobj(image_path.file, buffer)
    global automl
    test_preds, labels = automl.predict(local_image_path)
    os.remove(local_image_path)
    return {"prediction": test_preds, "labels": labels}



@app.post("/classification/train")
async def train_model(train_csv_path: UploadFile = File("upload your csv file")):
    local_csv_path = f"{train_csv_path.filename}"
    with open(local_csv_path, "wb") as buffer:
        shutil.copyfileobj(train_csv_path.file, buffer)
    global automl
    automl = AutoML()
    automl.model_fit(local_csv_path, "classification")
    os.remove(local_csv_path)   
    return {"message": "Model training completed!"}

@app.post("/classification/predict")
async def predict_csv(test_csv_path: UploadFile = File("upload your csv file")):
    local_csv_path = f"{test_csv_path.filename}"
    with open(local_csv_path, "wb") as buffer:
        shutil.copyfileobj(test_csv_path.file, buffer)
    global automl
    prediction = automl.predict(local_csv_path)
    prediction = prediction.tolist()
    os.remove(local_csv_path)
    return {"prediction": prediction}


@app.post("/regression/train")
async def train_model(train_csv_path: UploadFile = File("upload your csv file")):
    local_csv_path = f"{train_csv_path.filename}"
    with open(local_csv_path, "wb") as buffer:
        shutil.copyfileobj(train_csv_path.file, buffer)
    global automl
    automl = AutoML()
    automl.model_fit(local_csv_path, "regression")
    os.remove(local_csv_path)
    return {"message": "Model training completed!"}

@app.post("/regression/predict")
async def predict_csv(test_csv_path: UploadFile = File("upload your csv file")):
    local_csv_path = f"{test_csv_path.filename}"
    with open(local_csv_path, "wb") as buffer:
        shutil.copyfileobj(test_csv_path.file, buffer)
    global automl
    prediction = automl.predict(local_csv_path)
    prediction = prediction.tolist()
    os.remove(local_csv_path)
    return {"prediction": prediction}   

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

