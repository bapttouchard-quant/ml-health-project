from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_absolute_error

df = load_diabetes(as_frame=True).frame

X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(random_state=42, max_depth=2)

model.fit(X_train, y_train)

depth_values = [1, 2, 3, 4, 5, 7, 10]
for depth in depth_values:
    score = cross_val_score(DecisionTreeRegressor(max_depth=depth, random_state=42), X, y, cv=5, scoring="neg_mean_absolute_error")

    result = -score
    result_mean, result_std = result.mean(), result.std()
    print(depth, result_mean, result_std)
train_prediction = model.predict(X_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
train_mae = mean_absolute_error(y_train, train_prediction)

#print(train_prediction) 
#print ("Train MAE", train_mae)
#print("MAE", mae)

#print(predictions)

print(export_text(model, feature_names=list(X.columns)))