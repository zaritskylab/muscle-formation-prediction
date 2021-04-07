import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns


# Load the dataset
encoded_time_df = pd.read_pickle("../data/encoded cells & time stamps - diff.pkl")


# Split the data into training/testing sets
train, test = train_test_split(encoded_time_df, test_size=0.2)
encoded_X_train = train.drop("time", 1)
time_y_train = train['time']
encoded_X_test = test.drop("time", 1)
time_y_test = test['time']

# Create linear regression object
linear_reg = linear_model.LinearRegression()

# Train the model using the training sets
linear_reg.fit(encoded_X_train, time_y_train)

# Make predictions using the testing set
time_y_pred = linear_reg.predict(encoded_X_test)

# The coefficients
print('Coefficients: \n', linear_reg.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(time_y_test, time_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(time_y_test, time_y_pred))

#Plot outputs
# plt.scatter(encoded_X_test, time_y_test, color='black')
plt.show()
plt.scatter(time_y_test, time_y_pred, color='blue', linewidth=3)
plt.xlabel("real time")
plt.ylabel("prediction")
plt.title("real time over predicted time")
plt.show()
plt.hist(time_y_test - time_y_pred, color='green')
plt.title("histogram")
plt.xticks(())
plt.yticks(())
plt.show()


# predict on single cell:
encoded_time_df_single = encoded_time_df.head(404)
encoded_X_single = encoded_time_df_single.drop("time", 1)
time_y_single = encoded_time_df_single['time']
time_pred_single = linear_reg.predict(encoded_X_single)

# The coefficients
print('Coefficients: \n', linear_reg.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(time_y_test, time_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(time_y_test, time_y_pred))

#Plot outputs
# plt.scatter(encoded_X, time_y, color='black')
plt.show()
plt.scatter(time_y_single, time_pred_single, color='blue', linewidth=3)
plt.xlabel("real time")
plt.ylabel("prediction")
plt.title("real time over predicted time - single cell")
# plt.xlim(0, 110, 50)
# plt.ylim(0, 500, 50)
plt.show()
plt.hist(time_y_single - time_pred_single, color='green')
plt.title("histogram - single cell")
plt.xticks(())
plt.yticks(())
plt.show()