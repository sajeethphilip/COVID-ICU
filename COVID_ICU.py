import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import configparser
import joblib

# Load configuration file
config = configparser.ConfigParser()
config.read('ICU_Admin.conf')

# Read input_features and categorical columns from the configuration file
input_features = config['Config']['InputFeatures'].split(',')
categorical_columns = config['Config']['CategoricalColumns'].split(',')

# Load the data from icu.csv and nonicu.csv
icu_data = pd.read_csv('icu.csv')
nonicu_data = pd.read_csv('nonicu.csv')

# Concatenate the two datasets vertically
merged_data = pd.concat([icu_data, nonicu_data], ignore_index=True)

# Handle missing values by dropping rows with any missing values
merged_data = merged_data.dropna()

# Define the input features and target variable
X = merged_data[input_features]
y = (merged_data["ADMISSION"] == "ICU").astype(int)  # Convert ADMISSION to binary target (ICU: 1, NON ICU: 0)

# Separate numerical and categorical features
numerical_features = [col for col in input_features if col not in categorical_columns]
categorical_features = [col for col in input_features if col in categorical_columns]

# Apply one-hot encoding to categorical features
encoder = OneHotEncoder(sparse_output=False)  # Set sparse_output to False
X_categorical_encoded = encoder.fit_transform(X[categorical_features])

# Standardize numerical features
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X[numerical_features])

# Combine encoded categorical and scaled numerical features
X_processed = np.hstack((X_categorical_encoded, X_numerical_scaled))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train.to_numpy())
X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(y_test.to_numpy())

# Define a simple Fully Connected Neural Network using PyTorch
class FCNN(nn.Module):
    def __init__(self, input_size):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Output layer for binary classification with 2 classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = nn.functional.softmax(self.fc3(x), dim=1)  # Softmax activation for multi-class classification
        return x

# Initialize the model and optimizer
input_size = X_train_tensor.shape[1]
model = FCNN(input_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # Cross Entropy Loss for multi-class classification

# Early stopping parameters
best_accuracy = 0.0
patience = 50  # Number of epochs without improvement to wait before early stopping
counter = 0  # Counter to track patience

# Training loop
epochs = 1000  # You can adjust the number of epochs as needed
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor.long())  # Use long() to match target data type

    loss.backward()
    optimizer.step()

    # Evaluation on the test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs, dim=1)

    accuracy = accuracy_score(y_test, predicted)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Test Accuracy: {accuracy * 100:.2f}%")

    # Check for early stopping based on accuracy improvement above 95%
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        counter = 0
        # Save the best model
        joblib.dump(model, 'ICU_admin_best.pkl')
    else:
        if accuracy > 0.95:
            counter += 1
            if counter >= patience:
                print("Early stopping: No improvement in accuracy.")
                break

# Load the best model for subsequent training or inference
best_model = joblib.load('ICU_admin_best.pkl')
# Predict on the test data using the best model
best_model.eval()
with torch.no_grad():
    test_outputs = best_model(X_test_tensor)
    _, predicted = torch.max(test_outputs, dim=1)

test_accuracy = accuracy_score(y_test, predicted)
print(f"Test Accuracy using Best Model: {test_accuracy * 100:.2f}%")

# Load the original test data (before preprocessing)
original_test_data = merged_data.loc[y_test.index]

# Create a DataFrame for the output
output_data = original_test_data.copy()

# Add the "Prediction" column as categorical with class labels
output_data["Prediction"] = predicted
output_data["Prediction"] = output_data["Prediction"].apply(lambda x: "ICU" if x == 1 else "NON ICU")

# Add the "Confidence" column with normalized confidence scores
confidence_scores = nn.functional.softmax(test_outputs, dim=1).numpy()
output_data["Confidence ICU"] = confidence_scores[:, 1]  # Confidence for ICU class
output_data["Confidence NON ICU"] = confidence_scores[:, 0]  # Confidence for NON ICU class

# Save the output data to a CSV file
output_data.to_csv("output_basic_data.csv", index=False)
