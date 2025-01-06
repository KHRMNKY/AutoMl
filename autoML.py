import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import torch
import torchvision.models as models
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
import os
from torchvision.transforms import InterpolationMode


from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


class AutoML:
    def __init__(self):
        
        self.classification_models = {
            'random_forest': RandomForestClassifier(),
            'svm': SVC(),
            'decision_tree': DecisionTreeClassifier()
        }
        
        self.regression_models = {
            'random_forest': RandomForestRegressor(),
            'svr': SVR(),
            'decision_tree': DecisionTreeRegressor()
        }
        
                
        self.vision_classification_models = {
            'resnet18': models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
            'vgg16': models.vgg16(weights=models.VGG16_Weights.DEFAULT),
            'mobilenet': models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        }
        
        
        self.transform = transforms.Compose([
        transforms.Resize(232, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.task = None
        self.model = None
        self.model_name = None
        self.score = None
        self.encoder = None
        self.scaler = None

    def preprocess(self, X):
        df = pd.read_csv(X)
        df.columns = df.columns.astype(str)
        X_df = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        categorical_columns = X_df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            encoder = OneHotEncoder(drop='first', sparse_output=False)
            encoded_columns = encoder.fit_transform(X_df[categorical_columns])
            encoded_columns_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(categorical_columns))
            X_df = X_df.drop(columns=categorical_columns)
            X_processed = pd.concat([X_df, encoded_columns_df], axis=1)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_processed)
            return X_scaled, y, encoder, scaler

        else:
            X_scaled = StandardScaler().fit_transform(X_df)
            return X_scaled, y, None, None

        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            y = y.astype(int)

        return X, y

    def vision_preprocess(self, imgs_data_path):
        dataset = ImageFolder(imgs_data_path, transform=self.transform)
        torch.manual_seed(42)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
        

        classes = dataset.classes
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        
        return train_loader, test_loader, classes
    
    
    
    def model_fit(self, X, task = "classification", lr = None, epochs = None):
        model_name_list = []
        model_list = []
        scores_list = []
        
        
        if task == "classification":
            X, y = self.preprocess(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 )  
            models = self.classification_models
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc_score = accuracy_score(y_test, y_pred) 
                print(f"model: {name} ==>> accuracy : {acc_score} ")
                model_name_list.append(name)
                model_list.append(model)
                scores_list.append(acc_score)
                
                
                
        elif task == "vision_classification":
            train_loader, test_loader, classes = self.vision_preprocess(X)
            models = self.vision_classification_models
            for name, model in models.items():
                if name == "resnet18":
                    model.fc = nn.Linear(model.fc.in_features, len(classes))
                    model.to(self.device)
                elif name == "vgg16":
                    model.classifier[6] = nn.Linear(4096, len(classes))
                    model.to(self.device)
                elif name == "mobilenet":
                    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
                    model.to(self.device)
                
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, )
                
                model.train()
                num_epochs = epochs
                for epoch in range(num_epochs):
                    train_loss = 0
                    correct = 0
                    for images, labels in train_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        
                        preds = model(images)
                        loss = criterion(preds, labels)
                        train_loss+=loss.item()
                        correct += (preds.argmax(dim=1) == labels).sum().item()
                        
                        loss.backward()
                        optimizer.zero_grad()
                        optimizer.step()
                        
                    train_loss/=len(train_loader)
                    accuracy = 100 * correct / len(train_loader.dataset)
                    print(f"Model: {name}, Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f} Accuracy: {accuracy:.2f}%")

                model_list.append(model)
                model_name_list.append(name)
                scores_list.append(accuracy)
                
        elif task == "regression":
            X , y = self.preprocess(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 )
            models = self.regression_models
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2_scoree = r2_score(y_test, y_pred) 
                print(f"model: {name} ==>> r2_scoree : {r2_scoree} ")
                model_name_list.append(name) 
                model_list.append(model)
                scores_list.append(r2_scoree)
                
        else:   
            print("task should be classification or regression or vision_classification")
            
        self.score = max(scores_list)
        index = scores_list.index(self.score)
        self.model_name = model_name_list[index]
        self.model = model_list[index]   
        self.task = task
        
        print("-"*50)
        
        
            
    def predict(self, X):
        if self.model is None or self.score is None or self.task is None:
            print("you should fit your model before predicting !!!")
            return 
        
        if self.task == "vision_classification":
            dataset = ImageFolder(X, transform=self.transform)
            test_loader = DataLoader(dataset, batch_size=32, shuffle=False) 
            testLoss=0
            criterion = nn.CrossEntropyLoss()
            self.model.eval()
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                test_preds=self.model(imgs)
                loss= criterion(test_preds, labels)
                testLoss+=loss.item()
            testLoss/=len(test_loader)  
            print(f"best model: {self.model_name}, best model test Loss: {testLoss:.4f}")
            return test_preds.argmax(dim=1).cpu().numpy().tolist(), labels.cpu().numpy().tolist()
            
        else:
            X_df = pd.read_csv(X)
            X_df.columns = X_df.columns.astype(str)
            categorical_columns = X_df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_columns) > 0:
                encoder = OneHotEncoder(drop='first', sparse_output=False)
                encoded_columns = encoder.fit_transform(X_df[categorical_columns])
                encoded_columns_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(categorical_columns))
                X_df = X_df.drop(columns=categorical_columns)
                X = pd.concat([X_df, encoded_columns_df], axis=1)
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                y_pred = self.model.predict(X)
                print(f"best model: {self.model_name}, best model predictions: {y_pred}")
                
            else:
                X = X_df
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                y_pred = self.model.predict(X)
                print(f"best model: {self.model_name}, best model predictions: {y_pred}")
                
                
            return y_pred
