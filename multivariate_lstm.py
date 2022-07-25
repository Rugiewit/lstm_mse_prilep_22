

"""
dataset: macedonian exchange office
based on  https://youtu.be/tepxdcepTbY
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import mean_squared_error

#from datetime import datetime

#Read the csv file
df = pd.read_csv('./granit.csv')
#./komercialna.csv
#./alkaloid.csv
#./granit.csv
#./makpetrol.csv
#./makedonijaturist.csv
print(df.head()) #7 columns, including the Date.

#Separate dates for future plotting
train_dates = pd.to_datetime(df['date'])
print(train_dates.tail(15)) #Check last few dates.

#Variables for training
cols = list(df)[1:5]
#Date and volume columns are not used in training.
print(cols) #['max', 'min', 'open', 'close']

#New dataframe with only training data - 5 columns
df_for_training = df[cols].astype(float)

# df_for_plot=df_for_training.tail(5000)
# df_for_plot.plot.line()

#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)


#As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
#In this example, the n_features is 5. We will make timesteps = 14 (past days data used for training).

#Empty lists to be populated using formatted training data
trainX = []
trainY = []

n_future = 1  # Number of months we want to look into the future based on the past months.
n_past = 5  # Number of past months we want to use to predict the future.

#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (180, 5)
#12823 refers to the number of data points and 5 refers to the columns (multi-variables).
for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

# define the Autoencoder model

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()


# fit the model
history = model.fit(trainX, trainY, epochs=15, batch_size=4, validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

#Remember that we can only predict one month in future as our model needs 5 variables
#as inputs for prediction. We only have all 5 variables until the last month in our dataset.
n_past = 12
n_months_for_prediction=12  #let us predict past months 

predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_months_for_prediction, freq='M').tolist()
print(predict_period_dates)

#Make prediction
prediction = model.predict(trainX[-n_months_for_prediction:]) #shape = (n, 1) where n is the n_months_for_prediction

#Perform inverse transformation to rescale back to original range
#Since we used 5 variables for transform, the inverse expects same dimensions
#Therefore, let us copy our values 5 times and discard them after inverse transform
prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]


# Convert timestamp to date
forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())

df_forecast = pd.DataFrame({'date':np.array(forecast_dates), 'open':y_pred_future})
df_forecast['date']=pd.to_datetime(df_forecast['date'])
print(df_forecast['date'])

original = df[['date', 'open']]
original['date']=pd.to_datetime(original['date'])
print(original['date'])

#mse = mean_squared_error(original[col], df_forecast[col])
#print(mse)

original = original.loc[original['date'] >= '2020-06-01']
    
sns.lineplot(original['date'], original['open'])
sns.lineplot(df_forecast['date'], df_forecast['open'])


