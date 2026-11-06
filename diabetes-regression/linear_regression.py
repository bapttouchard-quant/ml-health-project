from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
import pandas as pd



df = load_diabetes(as_frame=True).frame

X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

#print(predictions, mae)

#scores = cross_val_score(LinearRegression(), X, y, cv=5, scoring="neg_mean_absolute_error")
scores = cross_val_score(Ridge(alpha=0.1), X, y, cv=5, scoring="neg_mean_absolute_error")
mae_folds = -scores
mae_std, mae_mean = mae_folds.std(), mae_folds.mean()

baseline_mae = (y_test - y_train.mean()).abs().mean()
print(-scores.mean())
#print(df[["s1", "s2"]].corr())
#print(mae, baseline_mae)
