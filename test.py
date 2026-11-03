import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

model = joblib.load("iris_model.pkl")
test_df = pd.read_csv("Iris.csv")

test_df = test_df.drop("Id",axis=1)

X = test_df.drop("Species", axis = 1)
y = test_df["Species"]
pred = model.predict(X)

acc = accuracy_score(pred, y)
print(acc)

assert len(pred)==len(y)
assert acc >= 0.99
print("All tests passed")