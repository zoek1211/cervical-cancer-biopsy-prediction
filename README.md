# cervical-cancer-biopsy-prediction

- This project implements and compares machine learning classification models to classify if patients need a biopsy based on cervical cancer risk factors.
- The dataset used is cervical cancer dataset from Kaggle, containing multiple attributes related to patient history and test results. 

## Project Overview

- The goal is to classify whether a cervical biopsy is needed (Biopsy target variable) using various patient features such as smoking habits, contraceptive use, and STD history.
- Handling missing data, scaling features, and addressing class imbalance are key steps in the pipeline.

## Dataset

- [Dataset Used](https://www.kaggle.com/datasets/loveall/cervical-cancer-risk-classification)


- File: cervical_cancer.csv

- Contains clinical and behavioral features of patients related to cervical cancer risk factors.

- Missing values represented as '?' are converted to NaN and handled during preprocessing.

## Data Preprocessing

- Replace '?' with NaN for missing data.

- Convert all columns to numeric types.

- Missing values imputed using either mean substitution or K-Nearest Neighbors Imputation (KNNImputer).

- Features scaled using StandardScaler for normalization.

- Class imbalance addressed using Synthetic Minority Over-sampling Technique (SMOTE).

## Exploratory Data Analysis (EDA)

- Displayed data shape and summary info.

- Visualized correlation matrix using a heatmap.

- Plotted histograms of all features.

- Calculated counts and percentages of boolean (categorical) features related to patient history and diagnosis.

## Machine Learning Models

- Three classification algorithms were implemented and evaluated:

### K-Nearest Neighbors (KNN)
- Missing data imputed using KNNImputer.

- Grid search used for hyperparameter tuning (n_neighbors, weights, metric).

- SMOTE applied to training data for class balancing.

- Performance metrics: accuracy, precision, recall, classification report, confusion matrix.

### Decision Tree
- Used entropy criterion with max depth and minimum samples constraints.

- Class weights adjusted to penalize minority class misclassification.

- SMOTE applied to training data.

- Decision tree visualized using plot_tree.

### Naive Bayes
- Missing values imputed using mean strategy.

- SMOTE applied to training data.

- Gaussian Naive Bayes classifier trained and evaluated.
 
