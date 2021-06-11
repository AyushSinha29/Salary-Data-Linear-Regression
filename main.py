import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('Salary_Data.csv')
x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y, test_size =0.2, random_state = 0 )
from sklearn.linear_model import LinearRegression
Lin = LinearRegression()
Lin.fit(train_x,train_y)
pred_y = Lin.predict(test_x)
np.concatenate((test_y, pred_y), axis=1)
np.abs(test_y - pred_y)
np.abs(test_y - pred_y).mean()
Lin.predict([[3.5]])
from sklearn import metrics
metrics.mean_absolute_error(test_y, pred_y)
metrics.mean_squared_error(test_y, pred_y)
