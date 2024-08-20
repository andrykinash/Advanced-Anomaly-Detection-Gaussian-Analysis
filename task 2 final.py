import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Normalization function for smooth distribution
def normalize_data(data, mean, std):
    normalized = (data - mean) / std
    normalized_df = pd.DataFrame(normalized, columns=data.columns)
    return normalized_df

# PDF function
def calculate_pdf(normalized_data, mean, std):
    pdf_values = {}
    for feature in normalized_data.columns:
        pdf_values[feature] = (1 / np.sqrt(2 * np.pi * std[feature]**2)) * \
                               np.exp(- (normalized_data[feature] - mean[feature])**2 / (2 * std[feature]**2))
    return pd.DataFrame(pdf_values)

# Calculate the percentile-based threshold for each feature based on their PDF values
def calculate_percentile_threshold(pdf_values, percentile=10): #edit percentile here
    thresholds = {}
    for feature in pdf_values.columns:
        thresholds[feature] = np.percentile(pdf_values[feature], percentile)
    return thresholds

# Make anomaly decisions based on the calculated thresholds
def make_anomaly_decisions(pdf_values, thresholds):
    anomaly_flags = pd.DataFrame()
    for feature in pdf_values.columns:
        anomaly_flags[feature] = pdf_values[feature] < thresholds[feature]
    anomaly_flags['total_flags'] = anomaly_flags.sum(axis=1)
    anomaly_flags['anomaly'] = anomaly_flags['total_flags'] > 0
    return anomaly_flags

# Load the training data
training_data_path = 'C:/Users/andry/Desktop/gausian/train/training-data.csv'
train_data = pd.read_csv(training_data_path)
features_to_use = train_data.columns.drop('time')
columns_to_exclude = ['cmp_b_s', 'f2_s'] #excluded for testing 
features_to_use = features_to_use.difference(columns_to_exclude)

# Calculations for training data
mean = train_data[features_to_use].mean()
std = train_data[features_to_use].std()

# Normalize training data and calculate PDF values to get the initial thresholds
normalized_train_df = normalize_data(train_data[features_to_use], mean, std)
pdf_train_values = calculate_pdf(normalized_train_df, mean, std)
thresholds = calculate_percentile_threshold(pdf_train_values)

# Read validation key
valid_key_path = 'C:/Users/andry/Desktop/gausian/validation/valid-key.txt'
valid_key = {}
with open(valid_key_path, 'r') as f:
    for line in f:
        file_name, label = line.strip().split(' ')
        valid_key[file_name] = int(label)

# Initialize performance metrics
TP = 0  # True Positive
TN = 0  # True Negative
FP = 0  # False Positive
FN = 0  # False Negative
false_positives = []
false_negatives = []

validation_folder_path = 'C:/Users/andry/gausian/midterm/validation/'
for file_name, actual_label in valid_key.items():
    # Load validation data
    validation_data_path = os.path.join(validation_folder_path, f"{file_name}.csv")
    validation_data = pd.read_csv(validation_data_path)
    
    # Normalize and calculate PDF values
    normalized_validation_df = normalize_data(validation_data[features_to_use], mean, std)
    pdf_values = calculate_pdf(normalized_validation_df, mean, std)
    
    # Make anomaly decisions
    anomaly_flags = make_anomaly_decisions(pdf_values, thresholds)
    
    # Calculate the percentage of abnormal rows
    total_rows = len(anomaly_flags)
    abnormal_rows = sum(anomaly_flags['anomaly'])
    abnormal_percentage = (abnormal_rows / total_rows) * 100
    
    print(f"File {file_name}: {abnormal_percentage}% abnormal rows. {1 if abnormal_percentage > 32 else 0}") #edit abnomal percentage here


    # Compare the model's decision with actual label
    predicted_label = 1 if abnormal_percentage > 32 else 0 #edit abnomal percentage here
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

# Output the performance metrics
print(f"True Positive: {TP}, True Negative: {TN}, False Positive: {FP}, False Negative: {FN}")
print(f"Precision: {Precision}, Recall: {Recall}, F1 Score: {F1_score}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")
