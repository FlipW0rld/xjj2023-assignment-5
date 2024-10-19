import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# ------------------- Data Preprocessing -------------------

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print("Missing values in train data:")
print(train_df.isnull().sum())
print("\nMissing values in test data:")
print(test_df.isnull().sum())

gender_mapping = {'Male': 1, 'Female': 0}
train_df['Gender'] = train_df['Gender'].map(gender_mapping)
test_df['Gender'] = test_df['Gender'].map(gender_mapping)

geography_train = pd.get_dummies(train_df['Geography'], prefix='Geography')
geography_test = pd.get_dummies(test_df['Geography'], prefix='Geography')

geography_train, geography_test = geography_train.align(geography_test, join='outer', axis=1, fill_value=0)

train_df = train_df.drop('Geography', axis=1)
test_df = test_df.drop('Geography', axis=1)

train_df = pd.concat([train_df, geography_train], axis=1)
test_df = pd.concat([test_df, geography_test], axis=1)

numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

means = train_df[numerical_features].mean()
stds = train_df[numerical_features].std()

train_df[numerical_features] = (train_df[numerical_features] - means) / stds
test_df[numerical_features] = (test_df[numerical_features] - means) / stds

test_ids = test_df['id']

train_df = train_df.drop(['CustomerId', 'Surname', 'id'], axis=1)
test_df = test_df.drop(['CustomerId', 'Surname', 'id'], axis=1)

X = train_df.drop('Exited', axis=1).values
y = train_df['Exited'].values
X_test = test_df.values

# ------------------- KNN Implementation -------------------

def euclidean_distance(x1, x2):
    """
    Calculate the Euclidean distance between two vectors.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

def get_neighbors(X_train, y_train, test_instance, k):
    """
    Find the k nearest neighbors of the test_instance.
    """
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], test_instance)
        distances.append((dist, y_train[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    return neighbors

def predict_probability(X_train, y_train, test_instance, k):
    """
    Predict the probability of 'Exited' for the test_instance.
    """
    neighbors = get_neighbors(X_train, y_train, test_instance, k)
    output_values = [neighbor[1] for neighbor in neighbors]
    probability = sum(output_values) / k
    return probability

# ------------------- Cross-Validation -------------------

from sklearn.model_selection import KFold

k_values = range(1, 21)
mean_auc_scores = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("Starting cross-validation to find the best k...")

for k in k_values:
    auc_scores = []
    print(f"Evaluating k={k}")
    for train_index, val_index in kf.split(X):
        X_train_cv, X_val_cv = X[train_index], X[val_index]
        y_train_cv, y_val_cv = y[train_index], y[val_index]

        y_val_preds = []
        for test_instance in X_val_cv:
            prob = predict_probability(X_train_cv, y_train_cv, test_instance, k)
            y_val_preds.append(prob)

        auc = roc_auc_score(y_val_cv, y_val_preds)
        auc_scores.append(auc)

    mean_auc = np.mean(auc_scores)
    mean_auc_scores.append(mean_auc)
    print(f"k={k}, Mean AUC-ROC: {mean_auc}")

best_k_index = np.argmax(mean_auc_scores)
best_k = k_values[best_k_index]
print(f"\nBest k: {best_k} with Mean AUC-ROC: {mean_auc_scores[best_k_index]}")

# ------------------- Prediction on Test Data -------------------

predictions = []

print("\nStarting predictions with best k...")

for index in range(len(X_test)):
    test_instance = X_test[index]
    prob = predict_probability(X, y, test_instance, best_k)
    sample_id = test_ids.iloc[index]
    predictions.append((sample_id, prob))
    if (index + 1) % 100 == 0 or index == len(X_test) - 1:
        print(f"Processed {index+1}/{len(X_test)} instances.")

print("Predictions completed.")

# ------------------- Generate Submission File -------------------

submission_df = pd.DataFrame(predictions, columns=['id', 'Exited'])
submission_df.to_csv('submission.csv', index=False)

print(f"\nSubmission file 'submission.csv' has been generated using k={best_k}.")
