import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = {
    "Experience": [1,2,3,4,5,6,7,8,9,10],
    "Salary": [2500,3000,3500,4000,4500,5200,5800,6500,7000,7800]
}

df = pd.DataFrame(data)

print(df)

#print(df.head())
#print(df.info())
#print(df.describe())

# df.head()
# df.info()
# df.describe()


plt.scatter(df["Experience"], df["Salary"])
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("experience vs salairy ")
plt.show()


X = df[["Experience"]]
y = df[["Salary"]]
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score
r2 =r2_score(y_test, y_pred)
print( "R2 Score:",r2)

print("Real salaries:", y_test.values)
print("pridicted salaries :", y_pred)

plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction Model")
plt.show()

plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction Model")
plt.show()

experience_new = [[12]]
salary_pred = model.predict(experience_new)
print("predicted salary for 12 yers experience :", salary_pred[0])
