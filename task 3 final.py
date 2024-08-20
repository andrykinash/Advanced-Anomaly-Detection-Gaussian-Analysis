import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to calculate the multivariate Gaussian PDF
def multivariate_gaussian_pdf(X, mu, cov):
    n = X.shape[1]
    diff = (X - mu).values
    return np.exp(-0.5 * np.sum(diff @ np.linalg.inv(cov) * diff, axis=1)) / np.sqrt((2 * np.pi)**n * np.linalg.det(cov))

# Function to make anomaly decisions
def make_anomaly_decisions(pdf_values, threshold):
    anomaly_flags = pdf_values < threshold
    return anomaly_flags

# Normalization function
def normalize_data(data, mean, std):
    return (data - mean) / std

# Reading training data
training_data_path = 'C:/Users/andry/Desktop/gausian/train/training-data.csv'
train_data = pd.read_csv(training_data_path)
features_to_use = train_data.columns.drop('time')

# Excluding some columns for testing
columns_to_exclude = []
features_to_use = features_to_use.difference(columns_to_exclude)

# Calculate mean and standard deviation for training data
mean = train_data[features_to_use].mean()
std = train_data[features_to_use].std()

# Normalize the training data
normalized_train_data = normalize_data(train_data[features_to_use], mean, std)

# Calculate the covariance matrix from the normalized training data
cov_matrix = normalized_train_data.cov()

# Calculate PDF values for normalized training data
pdf_train_values = multivariate_gaussian_pdf(normalized_train_data, np.zeros(len(mean)), cov_matrix)
threshold = np.percentile(pdf_train_values, 5)  # Using 5th percentile as the threshold

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
    
    # Normalize the validation data using the mean and std from training data
    normalized_validation_data = normalize_data(validation_data[features_to_use], mean, std)
    
    # Calculate PDF values for normalized validation data
    pdf_values = multivariate_gaussian_pdf(normalized_validation_data, np.zeros(len(mean)), cov_matrix)
    
    # Make anomaly decisions
    anomaly_flags = make_anomaly_decisions(pdf_values, threshold)
    
    # Calculate the percentage of abnormal rows
    total_rows = len(anomaly_flags)
    abnormal_rows = sum(anomaly_flags)
    abnormal_percentage = (abnormal_rows / total_rows) * 100
    
    # Output abnormal_percentage and label for this file
    print(f"File {file_name}: {abnormal_percentage:.2f}% abnormal rows. {'Anomaly' if abnormal_percentage > 25 else 'Normal'}")
    
    # Compare the model's decision with the actual label
    predicted_label = 1 if abnormal_percentage > 25 else 0
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
Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
F1_score = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0

# Output the performance metrics
print(f"True Positive: {TP}, True Negative: {TN}, False Positive: {FP}, False Negative: {FN}")
print(f"Precision: {Precision:.4f}, Recall: {Recall:.4f}, F1 Score: {F1_score:.4f}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")

# Testing phase on unseen data
test_file_names = [f"{i}.csv" for i in range(58)]
test_folder_path = 'C:/Users/andry/Desktop/gausian/test'

test_results = []

for test_file_name in test_file_names:
    # Load test data
    test_data_path = os.path.join(test_folder_path, test_file_name)
    test_data = pd.read_csv(test_data_path)
    
    # Normalize the test data using the mean and std from training data
    normalized_test_data = normalize_data(test_data[features_to_use], mean, std)
    
    # Calculate PDF values for test data
    pdf_values = multivariate_gaussian_pdf(normalized_test_data, np.zeros(len(mean)), cov_matrix)
    
    # Make anomaly decisions
    anomaly_flags = make_anomaly_decisions(pdf_values, threshold)
    
    # Calculate the percentage of abnormal rows
    total_rows = len(anomaly_flags)
    abnormal_rows = sum(anomaly_flags)
    abnormal_percentage = (abnormal_rows / total_rows) * 100
    
    # Decide the label based on abnormal_percentage
    predicted_label = 1 if abnormal_percentage > 25 else 0  # Using the same threshold as validation
    
    # Append the results to the list
    test_results.append((test_file_name, predicted_label))

# Output the results for the test phase
for file_name, label in test_results:
    print(f"{file_name} {label}")
