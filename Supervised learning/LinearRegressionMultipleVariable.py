import pandas as pd
import math
from sklearn import linear_model

df = pd.read_csv('homeprices1.csv')

median_bedrooms = math.floor(df.bedrooms.median())
df.bedrooms = df.bedrooms.fillna(median_bedrooms)
print(df)

# create the model and train it.
model = linear_model.LinearRegression()

model.fit(df[['area', 'bedrooms', 'age']], df.price)

# predict the value according to the order which we specified during training

predicted = model.predict([[2500, 4, 5]])
print(predicted)

# finding the co-efficent it will give 3 values due to 3 independent Variables
# with 3 coefficent values by their values we can find which parameter matters more
coefficent = model.coef_
print(coefficent)

intercept = model.intercept_
print(intercept)

# Internal part happening inside the linear regression prediction
price = 137.25 * 2500 + -26025 * 4 + -6825 * 5 + intercept
print(price)

# Both predicted and price is same

