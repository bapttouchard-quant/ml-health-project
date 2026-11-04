from sklearn.datasets import load_diabetes

data = load_diabetes(as_frame=True)
df = data.frame

average = df["target"].mean()
mae = abs(average - df["target"]).mean()

#print(data.DESCR)
#print(df.head())

print(mae)
#“If we predict 152 for all 442 patients, we’re off by about 66 points on average.”