import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import joblib
import pickle

# Read the csv file
df = pd.read_csv('homeprices.csv')
# Create the model
model = linear_model.LinearRegression()
# Train the model
model.fit(df[['area']], df.price)

predict = model.predict([[5000]])
print(predict)

print(model.coef_)

# saving throgh pickle
# write the file
with open('picklemodel',"wb") as file:
    pickle.dump(model, file)
# read the file and store it inside the variable
with open('picklemodel', "rb") as file:
    data = pickle.load(file)

print(data.coef_)

# saving the file in the more preserved way if joblib
# write the file
joblib.dump(model, 'joblib_model')
# read the file
job = joblib.load('joblib_model')

print(job.coef_)