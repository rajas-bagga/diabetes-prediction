import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from imblearn.over_sampling import SMOTE

# Load and clean data
data = pd.read_csv("diabetes_prediction_dataset.csv")
data = data[data['smoking_history'] != 'ever']
data = data[data['smoking_history'] != 'not current']
data = data[data['smoking_history'] != 'No Info']


# # Balance the dataset
# df_majority = data[data['diabetes'] == 0]
# df_minority = data[data['diabetes'] == 1]
# df_majority_undersampled = df_majority.sample(n=len(df_minority), random_state=42)
# data = pd.concat([df_majority_undersampled, df_minority], axis=0).reset_index(drop=True)


# Encode categorical variables
data = pd.get_dummies(data, columns=['smoking_history'], dtype=int)
data["gender"] = data["gender"].map({"Male": 0, "Female": 1})

data = data.dropna()


# Split features and target
X = data.drop("diabetes", axis=1)
Y = data["diabetes"]

feature_names = X.columns

smote = SMOTE(random_state=42)
X, Y = smote.fit_resample(X, Y)

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=feature_names)

# Split into train/test
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.25)


# Train model
model = RandomForestClassifier()

param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [9, 12, 15],
    'class_weight': ['balanced'],
}

print("Training model...")
grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(xtrain, ytrain)
print("Model trained!")
print("Best parameters found: ", grid_search.best_params_)
print("Best score found: ", grid_search.best_score_)
print(f"Accuracy Score: {grid_search.score(X, Y)}")

###### Running confusion matrix and classification report


print("Confusion Matrix:")
print(confusion_matrix(ytest, grid_search.predict(xtest)))
print("Classification Report:")
print(classification_report(ytest, grid_search.predict(xtest)))

with open('model.pkl', 'wb') as file:
    pickle.dump(grid_search, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)



# Input from user
# smoking_status = [0, 0, 0]

# gender = input("Enter your gender [male/female]: ").strip().lower()
# age = int(input("Enter your age: "))
# hypertension = int(input("Do you have hypertension? [0/1]: "))
# heart_disease = int(input("Do you have heart disease? [0/1]: "))
# bmi = float(input("Enter your BMI: "))
# smoking_history = input("Enter your smoking history [never/former/current]: ").strip().lower()
# blood_glucose_level = float(input("Enter your blood glucose level: "))
# glycated_hemoglobin = float(input("Enter your glycated hemoglobin: "))

# if smoking_history == "never":
#     smoking_status[0] = 1
# elif smoking_history == "former":
#     smoking_status[1] = 1
# elif smoking_history == "current":
#     smoking_status[2] = 1

# # Build input DataFrame
# dataframe = pd.DataFrame([{
#     "gender": 0 if gender == "male" else 1,
#     "age": age,
#     "hypertension": hypertension,
#     "heart_disease": heart_disease,
#     "bmi": bmi,
#     "HbA1c_level": glycated_hemoglobin,
#     "blood_glucose_level": blood_glucose_level,
#     "smoking_history_current": smoking_status[2],
#     "smoking_history_former": smoking_status[1],
#     "smoking_history_never": smoking_status[0],
# }])

# # Scale input and predict
# dataframe = pd.DataFrame(scaler.transform(dataframe), columns=X.columns)
# res = grid_search.predict(dataframe)

# print(f"Predicted Diabetes Status: {'Yes' if res[0] == 1 else 'No'}")
