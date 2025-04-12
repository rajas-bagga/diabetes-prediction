import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle

# Load and clean data
data = pd.read_csv("diabetes_prediction_dataset.csv")
data = data[data['smoking_history'] != 'ever']
data = data[data['smoking_history'] != 'not current']
data = data[data['smoking_history'] != 'No Info']

# Balance the dataset
# df_majority = data[data['diabetes'] == 0]
# df_minority = data[data['diabetes'] == 1]
# df_majority_undersampled = df_majority.sample(n=len(df_minority), random_state=42)
# data = pd.concat([df_majority_undersampled, df_minority], axis=0).reset_index(drop=True)

# Encode categorical variables
data = pd.get_dummies(data, columns=['smoking_history'], dtype=int)
data["gender"] = data["gender"].map({"Male": 0, "Female": 1})

# Split features and target
X = data.drop("diabetes", axis=1)
Y = data["diabetes"]

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Split into train/test
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.25)


# Train model
model = RandomForestClassifier(max_depth=9, n_estimators=100)

param_grid = {
    'n_estimators': [100, 500],
    'max_depth': [7, 9, 15],
    'criterion': ['gini', 'entropy'],
}

print("Training model...")
grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(xtrain, ytrain)
print("Model trained!")
print("Best parameters found: ", grid_search.best_params_)
print("Best score found: ", grid_search.best_score_)
print(f"Accuracy Score: {grid_search.score(X, Y)}")


# this code is for testing