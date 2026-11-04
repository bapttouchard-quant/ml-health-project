from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
import pandas as pd

df = load_diabetes(as_frame=True).frame

X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

ridge_coef = model.coef_

alpha_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
for alpha in alpha_values:
    score = cross_val_score(Ridge(alpha=alpha), X, y, cv=5, scoring="neg_mean_absolute_error")

    result = -score
    result_mean, result_std = result.mean(), result.std()
    print(alpha, result_mean, result_std)

