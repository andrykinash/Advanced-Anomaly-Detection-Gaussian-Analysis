import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to calculate the multivariate Gaussian PDF
def multivariate_gaussian_pdf(X, mu, cov):
    n = X.shape[1]
    diff = (X - mu).values
    return np.exp(-0.5 * np.sum(diff @ np.linalg.inv(cov) * diff, axis=1)) / np.sqrt((2 * np.pi)**n * np.linalg.det(cov)) #formula

# Function to make anomaly decisions
def make_anomaly_decisions(pdf_values, threshold):
    anomaly_flags = pdf_values < threshold
    return anomaly_flags

#test files
test_results = []
test_file_names = [f"{i}.csv" for i in range(58)]
test_folder_path = 'C:/Users/andry/Desktop/gausian/test'

# Reading training data
training_data_path = 'C:/Users/andry/Desktop/gausian/train/training-data.csv'
train_data = pd.read_csv(training_data_path)
features_to_use = train_data.columns.drop('time')

# Excluding some columns for testing
columns_to_exclude = []
features_to_use = features_to_use.difference(columns_to_exclude)

# Calculate mean and covariance matrix for training data
mean = train_data[features_to_use].mean()
cov_matrix = train_data[features_to_use].cov()

# Calculate PDF values for training data
pdf_train_values = multivariate_gaussian_pdf(train_data[features_to_use], mean, cov_matrix)
threshold = np.percentile(pdf_train_values, 5)  # using 5th percentile as threshold edit if needbe

# Initialize performance metrics
TP = 0
TN = 0
FP = 0
FN = 0
false_positives = []
false_negatives = []

# Read validation key
valid_key_path = 'C:/Users/andry/Desktop/gausian/validation/valid-key.txt'
valid_key = {}
with open(valid_key_path, 'r') as f:
    for line in f:
        file_name, label = line.strip().split(' ')
        valid_key[file_name] = int(label)

validation_folder_path = 'C:/Users/andry/Desktop/gausian/validation/'
for file_name, actual_label in valid_key.items():
    # Load validation data
    validation_data_path = os.path.join(validation_folder_path, f"{file_name}.csv")
    validation_data = pd.read_csv(validation_data_path)
    
    # Calculate PDF values for validation data
    pdf_values = multivariate_gaussian_pdf(validation_data[features_to_use], mean, cov_matrix)
    
    # Make anomaly decisions
    anomaly_flags = make_anomaly_decisions(pdf_values, threshold)
    
    # Calculate the percentage of abnormal rows
    total_rows = len(anomaly_flags)
    abnormal_rows = sum(anomaly_flags)
    abnormal_percentage = (abnormal_rows / total_rows) * 100
    
    # Output abnormal_percentage and label for this file
    print(f"File {file_name}: {abnormal_percentage}% abnormal rows. {1 if abnormal_percentage > 25 else 0}") #edit abnomal percentage here
    
    # Compare the model's decision with the actual label
    predicted_label = 1 if abnormal_percentage > 25 else 0 #edit abnomal percentage here
    if predicted_label == 1 and actual_label == 1:
        TP += 1
    elif predicted_label == 1 and actual_label == 0:
        FP += 1
        false_positives.append(file_name)
    elif predicted_label == 0 and actual_label == 1:
        FN += 1
        false_negatives.append(file_name)
    elif predicted_label == 0 and actual_label == 0:
        TN += 1

# Calculate Precision, Recall, and F1 score
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1_score = 2 * (Precision * Recall) / (Precision + Recall)

# Loop through each file in the test set
for test_file_name in test_file_names:
    # Load test data
    test_data_path = os.path.join(test_folder_path, test_file_name)
    test_data = pd.read_csv(test_data_path)
    
    # Calculate PDF values for test data
    pdf_values = multivariate_gaussian_pdf(test_data[features_to_use], mean, cov_matrix)
    
    # Make anomaly decisions
    anomaly_flags = make_anomaly_decisions(pdf_values, threshold)
    
    # Calculate the percentage of abnormal rows
    total_rows = len(anomaly_flags)
    abnormal_rows = sum(anomaly_flags)
    abnormal_percentage = (abnormal_rows / total_rows) * 100
    
    # Decide the label based on abnormal_percentage
    predicted_label = 1 if abnormal_percentage > 20 else 0  # Customize the threshold as needed
    
    # Append the results to the list
    test_results.append((test_file_name, predicted_label))

# Output the performance metrics
print(f"True Positive: {TP}, True Negative: {TN}, False Positive: {FP}, False Negative: {FN}")
print(f"Precision: {Precision}, Recall: {Recall}, F1 Score: {F1_score}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")
for file_name, label in test_results:
    print(f"{file_name} {label}")
