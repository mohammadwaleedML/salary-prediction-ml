import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

df = pd.read_csv("Salary_dataset1.csv")

df.dropna(inplace=True)

le= LabelEncoder()
df["education"] = le.fit_transform(df["education"])
df["job_role"] = le.fit_transform(df["job_role"])

x = df.drop("salary", axis=1)
y = df["salary"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("MAE:",mean_absolute_error(y_test,y_pred))
joblib.dump(model,"model.PK1")
print("Model Saved")

# print(df.head())


