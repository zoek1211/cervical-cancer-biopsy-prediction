import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set pandas display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Expand display width to fit all columns

# read dataset
df = pd.read_csv('cervical_cancer.csv')

# print the number of columns, rows
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')

df.head(10)

df.info()

# replace ? to nan
df = df.replace('?', np.nan)
df.head(10)

# count missing values
missing_data = df.isna().sum()
print(missing_data)

# change pd to numeric 
df = df.apply(pd.to_numeric)

# get basic statistics
basic_stats = df.describe()
print(basic_stats)

# replace nan with mean
df_mean = df.fillna(df.mean())

# remove nan 
df_removed = df.dropna()

# heat map 
plt.figure(figsize=(22, 22))
correlation_matrix = df_mean.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
# plt.savefig('correltaion_heatmap.png')
plt.show()

# histogram
histogram = df.hist(figsize= (20,20), color= 'blue')
plt.suptitle('Historgrams of columns')
# plt.savefig('histogram.png')
plt.show()

# calculate percentage of boolean values

# function to count, calculate percentage for boolean attributes
def calculate_boolean_percentage(column):
    counts = column.value_counts()
    total = column.notna().sum()
    percentages = (counts / total * 100).round(1)
    return pd.DataFrame({'Count': counts, 'Percentage': percentages})

boolean_columns = [
    'Smokes', 'Hormonal Contraceptives', 'IUD', 'STDs', 'STDs:condylomatosis',
    'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis', 
    'STDs:syphilis', 'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
    'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV', 'STDs:Hepatitis B',
    'STDs:HPV', 'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx',	'Hinselmann',	'Schiller',	
    'Citology',	'Biopsy'
]

# caluclate percentage for boolean columns
results = {}
for col in boolean_columns:
    if col in df.columns:
        results[col] = calculate_boolean_percentage(df[col])

# combine results into a summary table
summary_df = pd.concat(results, axis=1)

# Set pandas display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Expand display width to fit all columns

# display the summary table 
print(summary_df)

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
# #
# K-Nearest Neighbors Classification
# # 
# drop the target column
X = df.drop('Biopsy', axis=1)

# extract the target
y = df['Biopsy']

# Impute missing values using KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to handle class imbalance
smote = SMOTE(sampling_strategy='minority', k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define hyperparameter grid for KNN
param_grid = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Create GridSearchCV for KNN
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best parameters and the best KNN model
print("Best Parameters:", grid_search.best_params_)
best_knn = grid_search.best_estimator_

# Make predictions with the best model
y_pred = best_knn.predict(X_test)

print("KNN classification: ")

# Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

# Additional metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

#
# Decision Tree Classification
#
from sklearn.tree import DecisionTreeClassifier, plot_tree 

# drop the target column
X = df.drop('Biopsy', axis=1)

# extract the target
y = df['Biopsy']

# impute missing values
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data first
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE only on the training data
smote = SMOTE(sampling_strategy='minority', k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

dt = DecisionTreeClassifier(
    criterion='entropy', 
    random_state=42, 
    class_weight={0: 1, 1: 5}, 
    max_depth=5, 
    min_samples_split=10, 
    min_samples_leaf=5
)

# train decision tree classifier
dt.fit(X_train_resampled, y_train_resampled)

# predict
y_pred = dt.predict(X_test)

# evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

#print the output
print("Decision Tree classification:")
print(classification_report(y_test, y_pred))

print("confusion matrix: \n", confusion_matrix(y_test, y_pred))

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

selected_features = X.columns

# plot decision tree
plt.figure(figsize=(20, 10)) 
plot_tree(dt, feature_names=selected_features, class_names=['0', '1'], filled=True, rounded=True) 
plt.show() 

#
# Naive Bayes Classification
#
from sklearn.naive_bayes import GaussianNB

# drop the target column
X = df.drop('Biopsy', axis=1)

# extract the target
y = df['Biopsy']

# impute missing values
# imputer = KNNImputer(n_neighbors=5)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# split to train, test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the classes
smote = SMOTE(sampling_strategy='minority', k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# create navie bayes model
nb = GaussianNB()

# Adjust Naive Bayes with class_prior for better handling of class imbalance
# nb = GaussianNB(priors=[0.5, 0.5])

# train naive bayes model
nb.fit(X_train, y_train)

# predict using naive bayes
y_pred = nb.predict(X_test)

# evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Naive Bayes Classification: ")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("confusion matrix: \n", confusion_matrix(y_test, y_pred))