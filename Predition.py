from random import sample

import joblib
import pandas as pd

model = joblib.load("model.PK1")


# order: age, experience, education, job_role
sample = [[25, 3, 1, 0]]

feature_names =['age','experience','education','job_role']
sample_df = pd.DataFrame(sample,columns=feature_names)


prediction = model.predict(sample_df)

print("Predicted Salary:", prediction[0])