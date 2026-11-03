import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import joblib

df = pd.read_csv('Iris.csv')
df = df.drop("Id", axis=1)

x = df.drop("Species", axis=1)
y = df["Species"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

iris_model = LogisticRegression()
iris_model.fit(x_train, y_train)

y_pred = iris_model.predict(x_test)
print("predicted values for first sample:", y_pred[0])

accuracy = accuracy_score(y_pred, y_test)
print("accuracy of the model is:", accuracy)

joblib.dump(iris_model, 'iris_model.pkl')
print("model trained and saved successfully")