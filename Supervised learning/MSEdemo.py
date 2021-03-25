import numpy as np
# in-build method for mean square error in sklearn
from sklearn.metrics import mean_squared_error

actual_y =[1, 1, 2, 3, 1, 4]
predict_y = [1, 0.5, 2.5, 3, 5, 1.5]

mse = np.square(np.subtract(predict_y, actual_y)).mean()

print(mse)

# Through sklearn
mse_sklearn = mean_squared_error(actual_y, predict_y)
print(mse_sklearn)