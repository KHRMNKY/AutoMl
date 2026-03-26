from autoML import AutoML

# TEST 1: Classification with Tabular Data
print("TEST 1: Classification")
automl = AutoML()
automl.model_fit('data/Iris_train.csv', task='classification')
predictions = automl.predict('data/Iris_train.csv')
print(f"Best Model: {automl.model_name}")
print(f"Score: {automl.score}\n")

# TEST 2: Regression with Tabular Data
print("TEST 2: Regression")
automl = AutoML()
automl.model_fit('data/Iris_train.csv', task='regression')
predictions = automl.predict('data/Iris_train.csv')
print(f"Best Model: {automl.model_name}")
print(f"Score: {automl.score}\n")

# TEST 3: Vision Classification
print("TEST 3: Vision Classification")
automl = AutoML()
automl.model_fit('data/images', task='vision_classification', lr=0.001, epochs=10)
predictions = automl.predict('data/images')
print(f"Best Model: {automl.model_name}")
print(f"Score: {automl.score}\n")

# TEST 4: Predict Before Fit (Error Handling)
print("TEST 4: Predict Before Fit")
automl = AutoML()
result = automl.predict('data/Iris_train.csv')
print(f"Result: {result}")