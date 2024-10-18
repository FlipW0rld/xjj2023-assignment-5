import pandas as pd
import numpy as np

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

#----------------------------------

test_ids = test_df['id']


train_df = train_df.drop(['CustomerId', 'Surname', 'id'], axis=1)
test_df = test_df.drop(['CustomerId', 'Surname', 'id'], axis=1)

X_train = train_df.drop('Exited', axis=1)
y_train = train_df['Exited']

X_train = X_train.values
y_train = y_train.values
X_test = test_df.values

#----------------------------------

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

#----------------------------------

k = 5  

predictions = []

print("\nStarting predictions...")

for index in range(len(X_test)):
    test_instance = X_test[index]
    prob = predict_probability(X_train, y_train, test_instance, k)
    sample_id = test_ids.iloc[index]
    predictions.append((sample_id, prob))
    if (index + 1) % 100 == 0 or index == len(X_test) - 1:
        print(f"Processed {index+1}/{len(X_test)} instances.")

print("Predictions completed.")

#---------------------------------------

submission_df = pd.DataFrame(predictions, columns=['id', 'Exited'])

submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' has been generated.")
