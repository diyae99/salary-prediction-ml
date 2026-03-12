import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Age": [22, 24, 25, 27, 29, 31, 33, 35, 37, 40],
    "Education": ["Bachelor", "Bachelor", "Bachelor", "Master", "Master", "Master", "Master", "PhD", "PhD", "PhD"],
    "Hours_per_week": [35, 36, 38, 40, 40, 42, 43, 45, 45, 48],
    "Projects_done": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Certifications": [0, 0, 1, 1, 2, 2, 3, 3, 4, 5],
    "Salary": [2500, 3000, 3600, 4200, 4900, 5600, 6200, 7000, 7800, 8800]
}

df = pd.DataFrame(data)

# Encoding
df = pd.get_dummies(df, columns=["Education"])

print(df.head())

# Features and target
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

print("Real salaries:", y_test.values)
print("Predicted salaries:", y_pred)

# Graph: Salary vs Experience only
#plt.scatter(df["Experience"], y, color="blue")
#plt.xlabel("Experience")
#plt.ylabel("Salary")
#plt.title("Salary vs Experience")
#plt.show()

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print(coefficients)

plt.figure(figsize=(10, 5))
plt.bar(coefficients["Feature"], coefficients["Coefficient"])
plt.xticks(rotation=45)
plt.xlabel("Features")
plt.ylabel("Coefficient")
plt.title("Feature Influence on Salary")
plt.show()

# New prediction
new_person = pd.DataFrame({
    "Experience": [12],
    "Age": [45],
    "Hours_per_week": [50],
    "Projects_done": [12],
    "Certifications": [5],
    "Education_Bachelor": [0],
    "Education_Master": [1],
    "Education_PhD": [0]
})

salary_pred = model.predict(new_person)
print("Predicted salary for new person:", salary_pred[0])