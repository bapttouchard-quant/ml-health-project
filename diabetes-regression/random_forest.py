from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error

#Load the data
df = load_diabetes(as_frame=True).frame

X = df.drop(columns="target")
y = df["target"]

n_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500]
for n_estimators in n_values:
    score = cross_val_score(RandomForestRegressor(n_estimators=n_estimators, random_state=42), X, y, cv = 5, scoring="neg_mean_absolute_error")
    
    result = -score
    result_mean, result_std = result.mean(), result.std()
    print(n_estimators, result_mean, result_std)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

test_prediction = model.predict(X_test)
train_prediction = model.predict(X_train)
train_mae = mean_absolute_error(y_train, train_prediction)
test_mae = mean_absolute_error(y_test, test_prediction)

print("Train MAE:", train_mae)
print("Test MAE:", test_mae)