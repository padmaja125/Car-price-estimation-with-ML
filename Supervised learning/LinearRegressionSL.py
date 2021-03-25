import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('homeprices.csv')
print(df)

plt.scatter(df.area, df.price)
plt.xlabel('area(sq.ft)')
plt.ylabel('Price(rupees)')
plt.show()

# model is created
model = linear_model.LinearRegression()
# training the model with dependent and independent values
model.fit(df[['area']], df.price)

# predict the value
predict = model.predict([[5000]])

# intercept and coefficient
coefficient = model.coef_
intercept = model.intercept_
print('coefficient', coefficient, 'intercept', intercept)

# manual checking

dependent = coefficient * 5000 + intercept
print('dependent', dependent, 'predict', predict)

plt.scatter(df.area, df.price)
plt.xlabel('area(sq.ft)')
plt.ylabel('Price(rupees)')
plt.plot(df.area, model.predict(df[['area']]), color='red')
plt.show()

# Simple linear regression by passing through csv file and saving it inside the another csv
areas = pd.read_csv('areas.csv')
dataframe = model.predict(areas)
areas['price'] = dataframe

areas.to_csv('FinalList.csv', index=False)
