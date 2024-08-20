# Anomaly Detection With Gaussian Analysis Machine Learning

## Overview

This project focuses on detecting anomalies in a chemical process based on data collected from a Programmable Logic Controller (PLC). The PLC reads sensor and actuator values from various components involved in the process and calculates an updated state. This project utilizes statistical methods, including Independent Gaussian Analysis and Multivariate Gaussian Analysis, to identify abnormal patterns in the dataset.

## Datasets

- All information on the data being processed can be found in "Dataset_Explained.txt" file

## Methodology

### Independent Gaussian Analysis

- **Data Preprocessing**: 
  - Missing values in the dataset were handled by replacing NaNs with the mean value of their respective feature.
  - Certain columns (`cmp_b_s`, `f2_s`) were excluded due to their high variance, which could distort the anomaly detection process.
  
- **PDF Calculation**:
  - Each feature's Probability Density Function (PDF) was calculated using its mean and standard deviation.
  
- **Threshold Determination**:
  - Percentile-based thresholds were determined for each feature to flag anomalies.
  
- **Anomaly Detection**:
  - The model flags rows as anomalous if their PDF values fall below the threshold.
  - Majority voting across all features is used to determine whether a row is anomalous.

### Multivariate Gaussian Analysis

- **Covariance Matrix Calculation**:
  - A covariance matrix was computed from the training data to account for correlations between features.
  
- **Multivariate PDF Calculation**:
  - The multivariate Gaussian PDF was calculated for each row in the dataset.
  
- **Anomaly Detection**:
  - Anomalies were flagged based on a predefined threshold (e.g., 5th percentile of training PDF values).
  
- **Performance Evaluation**:
  - Metrics such as True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) were calculated to evaluate the model's performance.

## Results

### Independent Gaussian Analysis

- **Performance**:
  - After optimizing the thresholds, the best results for the validation set were:
    - **True Positive**: 15
    - **True Negative**: 8
    - **False Positive**: 0
    - **False Negative**: 0
    - **Precision**: 1.0
    - **Recall**: 1.0
    - **F1 Score**: 1.0

### Multivariate Gaussian Analysis

- **Performance**:
  - The results were:
    - **True Positive**: 11
    - **True Negative**: 8
    - **False Positive**: 0
    - **False Negative**: 4
    - **Precision**: 1.0
    - **Recall**: 0.73
    - **F1 Score**: 0.85

### Conclusion

- **Independent Gaussian Analysis** showed excellent performance with a perfect F1 score on the validation set, though there is a concern about overfitting.
- **Multivariate Gaussian Analysis** the performance suggests that Multivariate Gaussian Analysis may not be the optimal method for this particular dataset, possibly due to the inherent characteristics of the data that could influence the model's ability to accurately distinguish between normal and abnormal behavior.

### Visualization of the data
![all features file 1 v file 2 batch 1](https://github.com/user-attachments/assets/d47c25f3-7234-4f1c-884c-785d04fb6bce)
![all features file 1 v file 2 batch 2](https://github.com/user-attachments/assets/19413592-0749-48e2-92a8-f26738198e0d)
![all features file 1 v file 2 batch 3](https://github.com/user-attachments/assets/d5ac1751-4107-4015-b1f3-dd6f301522f5)
![all features file 1 v file 2 batch 4](https://github.com/user-attachments/assets/55df2d64-3e72-4f46-86cd-7b499d55e8c5)
![all features file 1 v file 2 batch 5](https://github.com/user-attachments/assets/3a95b161-2680-458d-9c15-5d28a287ff9e)
![all features file 1 v file 2 batch 6](https://github.com/user-attachments/assets/704a3eda-f0ef-4fd5-a63e-3afe7cb4004e)
![all features file 1 v file 2 batch 7](https://github.com/user-attachments/assets/6e925eff-dade-4d6e-9dae-9435693e98a1)
![cmp_c_s and cmp_c_d in file 1 vs training data](https://github.com/user-attachments/assets/f570952b-7b9f-4b9b-93ee-def889f3d78c)
![file 1 vs file 2](https://github.com/user-attachments/assets/01a9a2af-7440-4979-a0da-7ff91a942290)
![important feature comparison](https://github.com/user-attachments/assets/caf81cf3-8cd9-4392-9f66-cf353fd6931a)
![sensor vs desired](https://github.com/user-attachments/assets/5d048a25-bd01-47fc-b846-7c869991037b)
