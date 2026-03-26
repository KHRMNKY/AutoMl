# AutoML - Automated Machine Learning Platform

An intelligent, cloud-deployed machine learning platform that automates model selection and training for classification, regression, and computer vision tasks. Features both REST API and web UI interfaces.

**Live Demo**: [https://automl.kahramankaya.com/docs#/](https://automl.kahramankaya.com/docs#/)

---

## ✨ Features

- **Three ML Task Types**:
  - 📊 **Tabular Classification** - Automatic model selection from Random Forest, SVM, Decision Tree
  - 📈 **Regression** - Automatic selection from Random Forest, SVR, Decision Tree
  - 🖼️ **Vision Classification** - Deep learning models (ResNet18, VGG16, MobileNet v2) with transfer learning

- **Automatic Model Selection**: Trains multiple models and automatically selects the best performer
- **GPU Support**: CUDA-enabled Docker container for fast training
- **Dual Interfaces**: FastAPI REST endpoints + Streamlit web UI
- **Cloud Ready**: Fully containerized and deployed on cloud infrastructure
- **Flexible Input**: Supports CSV files and image directories
- **Pre-configured Transforms**: Optimized image preprocessing with augmentation

---

## Quick Start

### Installation

Clone the repository:
```bash
git clone https://github.com/KHRMNKY/AutoMl.git
cd AutoMl
```

Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📚 Usage

### Option 1: REST API

Start the FastAPI server:
```bash
uvicorn api:app --reload
```

The API will be available at `http://127.0.0.1:8000`

View interactive API documentation at `http://127.0.0.1:8000/docs`

### Option 2: Streamlit Web UI

Run the Streamlit interface:
```bash
streamlit run streamlit4.py
```

Access the UI at `http://localhost:8501`

---

## 📖 API Endpoints

### Vision Classification
- **POST** `/vision_classification/train` - Train vision model
  ```json
  {
    "train_folder": "/path/to/train/folder",
    "learning_rate": 0.001,
    "epochs": 10
  }
  ```

- **POST** `/vision_classification/predict` - Predict on images
  ```json
  {
    "image_path": "/path/to/image/folder"
  }
  ```

### Classification (CSV)
- **POST** `/classification/train` - Train classification model
  ```json
  {
    "train_csv_path": "/path/to/train.csv"
  }
  ```

- **POST** `/classification/predict` - Get predictions
  ```json
  {
    "test_csv_path": "/path/to/test.csv"
  }
  ```

### Regression (CSV)
- **POST** `/regression/train` - Train regression model
  ```json
  {
    "train_csv_path": "/path/to/train.csv"
  }
  ```

- **POST** `/regression/predict` - Get predictions
  ```json
  {
    "test_csv_path": "/path/to/test.csv"
  }
  ```

---

## 📁 Project Structure

```
AutoMl/
├── api.py                    # FastAPI backend server
├── autoML.py                 # Core AutoML class with model logic
├── streamlit4.py             # Streamlit web UI
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker containerization
├── LICENSE                   # Apache 2.0 License
├── README.md                 # This file
└── data/
    ├── Iris_train.csv        # Sample training data
    ├── Iris_test.csv         # Sample testing data
    └── images/               # Sample image datasets
        ├── apple_pie/
        ├── baby_back_ribs/
        └── baklava/
```

---

## 🔧 Configuration

### Vision Classification Parameters
- **Input Size**: 224×224 pixels
- **Batch Size**: 32
- **Train/Test Split**: 80/20
- **Augmentation**: TrivialAugmentWide with 31 magnitude bins
- **Normalization**: ImageNet standard (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Supported Models

**Classification/Regression (Tabular)**:
- Random Forest
- Support Vector Machine (SVM/SVR)
- Decision Tree

**Vision Classification**:
- ResNet18 (ImageNet pre-trained)
- VGG16 (ImageNet pre-trained)
- MobileNet v2 (ImageNet pre-trained)

---

## 🐳 Docker Deployment

Build the Docker image:
```bash
docker build -t automl:latest .
```

Run the container:
```bash
docker run -p 8081:8081 --gpus all automl:latest
```

The API will be available at `http://localhost:8081`


---

## 📊 Example Usage

### Using Python

```python
from autoML import AutoML

# Initialize AutoML
automl = AutoML()

# Train classification model
automl.model_fit('data/Iris_train.csv', task='classification')

# Make predictions
predictions = automl.predict('data/Iris_test.csv')
print(f"Best Model: {automl.model_name}")
print(f"Score: {automl.score}")
print(f"Predictions: {predictions}")
```


---

## 📦 Dependencies

- **FastAPI**: Web framework for building APIs
- **PyTorch**: Deep learning framework
- **Scikit-learn**: Machine learning algorithms
- **Streamlit**: Web UI framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Torchvision**: Computer vision utilities
- **Pillow**: Image processing

See `requirements.txt` for complete list with versions.

---

## 🌐 Deployment

This project is currently deployed at: **https://automl.kahramankaya.com/docs#/**

The API is hosted on cloud infrastructure with GPU support for accelerated training.

---

## 📝 Data Format

### For Classification/Regression
- **Format**: CSV files
- **Target**: Last column should be the target variable
- **Features**: All other columns are treated as features
- **Preprocessing**: Automatic handling of categorical variables and feature scaling

### For Vision Classification
- **Format**: Folder structure with class subfolders
- **Example**:
  ```
  train_data/
  ├── class1/
  │   ├── image1.jpg
  │   └── image2.jpg
  └── class2/
      ├── image3.jpg
      └── image4.jpg
  ```



---


