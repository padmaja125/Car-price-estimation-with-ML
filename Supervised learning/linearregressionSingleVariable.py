import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# read the file
df = pd.read_csv(r'C:\Data\Padhu\tutorial\sample files\csv\canada_per_capita_income.csv')

# create the model
model = linear_model.LinearRegression()

# train the model
model.fit(df[['year']], df.percapitalincome)

# predict the model
predict = model.predict([[2020]])
print(predict)

# check with formula
# coefficient
coefficient = model.coef_
# intercept
intercept = model.intercept_

predicted = coefficient * 2020 + intercept
print(predicted)


# show in graph
plt.scatter(df.year, df.percapitalincome)
plt.xlabel('Year')
plt.ylabel('Per Capital Income (US $)')
plt.plot(df.year, model.predict(df[['year']]), color='blue')
plt.show()