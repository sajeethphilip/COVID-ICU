# ICU Admission Prediction Using Machine Learning

This repository contains Python code for predicting the possibility of ICU admission based on conditions at the time of admission and the progress in the following days of COVID-19. The code preprocesses the data, trains a simple Fully Connected Neural Network (FCNN) using PyTorch, and saves the best model for subsequent use.

## Overview

This project aims to predict whether a patient will be admitted to the ICU or not based on various input features and conditions. It includes data preprocessing, model training, early stopping, and model evaluation.

## Getting Started

To get started with this project, follow these steps:

### Prerequisites

1. Install Python: Make sure you have Python 3.x installed on your system.

2. Install Required Packages: You can install the required packages using the following command:

   ```
   pip install numpy pandas torch scikit-learn joblib
   ```

### Usage

1. Configure Input Features:

   - Modify the `ICU_Admin.conf` configuration file to specify the input features and categorical columns you want to use for prediction.

2. Load and Preprocess Data:

   - Ensure that you have the data in CSV format for ICU admissions and non-ICU admissions. The code reads data from `icu.csv` and `nonicu.csv`.

3. Run the Script:

   - Execute the main script:

     ```
     python icu_admission_prediction.py
     ```

4. Model Training:

   - The code will preprocess the data, train a FCNN model, and perform early stopping based on accuracy improvement.

5. Save the Best Model:

   - The best model is saved to the `ICU_admin_best.pkl` file for subsequent use.

6. Model Evaluation:

   - The script evaluates the best model on the test data and calculates the test accuracy.

7. Output Data:

   - The prediction results, confidence scores, and other details are saved in a CSV file named `output_basic_data.csv`.

## Model

The code defines a simple Fully Connected Neural Network (FCNN) using PyTorch for binary classification. You can adjust the model architecture and hyperparameters as needed.

## Early Stopping

The script implements early stopping based on accuracy improvement. If the accuracy surpasses 95%, it checks for improvement over a specified number of epochs (patience) before stopping early.

## Author

- [Your Name] (Email: your.email@example.com)

## License

This code is provided under an open-source license. You are free to use and modify it for your own purposes.

Please feel free to reach out if you have any questions or need further assistance.

---

*Note: Ensure that you have the necessary permissions and rights to use the data for your specific application, especially if it is for research or commercial use.*
