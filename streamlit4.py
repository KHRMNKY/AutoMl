import streamlit as st
import os
from pathlib import Path
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000"

st.title("Automated Machine Learning UI")

task = st.selectbox("Select Task Type", ["classification", "regression", "vision_classification"])

if task == "vision_classification":
    st.subheader("Vision Classification: Upload Image Folders")

    train_folder = st.text_input("Training Dataset Path (Folder)")
    if train_folder and not os.path.isdir(train_folder):
        st.error("Please enter a valid folder path!")

    learning_rate = st.number_input("Learning Rate", value=0.001, step=0.0001, format="%.4f")
    epochs = st.number_input("Number of Epochs", value=10, step=1, min_value=1)

    if st.button("Train Model"):
        if train_folder and os.path.isdir(train_folder):
            response = requests.post(
                f"{API_URL}/vision_classification/train",
                data={
                    "train_folder": train_folder,
                    "learning_rate": learning_rate,
                    "epochs": epochs
                }
            )
            if response.status_code == 200:
                st.success("Model trained successfully!")
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
        else:
            st.error("Please select a valid training dataset folder!")

    test_folder = st.text_input("Testing Dataset Path (Folder)")
    if test_folder and not os.path.isdir(test_folder):
        st.error("Please enter a valid folder path!")

    if st.button("Predict"):
        if test_folder and os.path.isdir(test_folder):
            response = requests.post(
                f"{API_URL}/vision_classification/predict",
                data={"image_path": test_folder}
            )
            if response.status_code == 200:
                predictions = response.json()
                st.write("Predictions:")
                st.json(predictions)
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
        else:
            st.error("Please select a valid testing dataset folder!")

else:
    st.subheader("Provide Dataset Paths")

    train_csv_path = st.text_input("Training Dataset Path (CSV)")
    if train_csv_path and not os.path.isfile(train_csv_path):
        st.error("Please enter a valid file path!")

    if st.button("Train Model (CSV)"):
        if train_csv_path and os.path.isfile(train_csv_path):
            response = requests.post(
                f"{API_URL}/{task}/train",
                data={"train_csv_path": train_csv_path}
            )
            if response.status_code == 200:
                st.success("Model trained successfully!")
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
        else:
            st.error("Please enter a valid training CSV file path!")

    test_csv_path = st.text_input("Testing Dataset Path (CSV)")
    if test_csv_path and not os.path.isfile(test_csv_path):
        st.error("Please enter a valid file path!")

    if st.button("Predict (CSV)"):
        if test_csv_path and os.path.isfile(test_csv_path):
            response = requests.post(
                f"{API_URL}/{task}/predict",
                data={"test_csv_path": test_csv_path}
            )
            if response.status_code == 200:
                predictions = response.json()
                st.write("Predictions:")
                st.json(predictions)
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
        else:
            st.error("Please enter a valid testing CSV file path!")