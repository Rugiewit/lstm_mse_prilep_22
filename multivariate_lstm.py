

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
companies = ['Комерцијална банка Скопје', 'Алкалоид Скопје','Гранит Скопје','Макпетрол Скопје','Македонијатурист Скопје']
c_index=0
#Read the csv file
df = pd.read_csv(companies[c_index]+'.csv')
#./komercialna.csv
#./alkaloid.csv
#./granit.csv
#./makpetrol.csv
#./makedonijaturist.csv
print(df.head()) #5 columns, including the Date.

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
n_past = 3 # Number of past months we want to use to predict the future.

#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (180, 5)
for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future,0])


trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

# define the Autoencoder model

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='sgd', loss='mse')
model.summary()


# fit the model
history = model.fit(trainX, trainY, epochs=100, batch_size=4, validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
 
plt.savefig('figures/'+companies[c_index]+"_loss.pdf")
plt.show()
plt.clf()

#Remember that we can only predict one month in future as our model needs 5 variables
#as inputs for prediction. We only have all 5 variables until the last month in our dataset.
# generate the multi-step forecasts
n_months_future = 6
prediction_future = []

x_pred = trainX[-1:,:,:]  # last observed input sequence
y_pred = trainY[-1]         # last observed target value
#Make prediction
#for i in range(n_months_for_prediction):
    # feed the last forecast back to the model as an input


for i in range(n_months_future):
    a=x_pred[:, 1:, :]    
    b= y_pred.reshape(1,1,1)
    b = np.repeat(b, df_for_training.shape[1], axis=-1)

    x_pred = np.append(a, b, axis=1)
    # generate the next forecast
    y_pred = model.predict(x_pred) 
    flat = y_pred.flatten()
    # save the forecast
    prediction_future.append(flat[0])
    

    
#Perform inverse transformation to rescale back to original range
#Since we used 5 variables for transform, the inverse expects same dimensions
#Therefore, let us copy our values 5 times and discard them after inverse transform
    # transform the forecasts back to the original scale
predict_period_dates = pd.date_range(list(train_dates)[-1], periods=n_months_future, freq='MS').tolist()
prediction_future = np.array(prediction_future).reshape(-1, 1)
prediction_copies = np.repeat(prediction_future, df_for_training.shape[1], axis=-1)
inverse_prediction = scaler.inverse_transform(prediction_copies)[:]
y_pred_future = pd.DataFrame(inverse_prediction, columns=cols)

# Convert timestamp to date
forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())
    


for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())

df_forecast=pd.DataFrame({'date':np.array(forecast_dates)})
df_forecast['date']=pd.to_datetime(df_forecast['date'], format = '%Y-%m-%d')
y_pred_future['date'] = df_forecast

error_months = 12
date_from = list(train_dates)[-error_months]


original = df[['date', 'max','min','open','close']]
original = df.copy()
original['date']=pd.to_datetime(original['date'])
original = original.loc[original['date'] >= date_from]


error_prediction = model.predict(trainX[-error_months:]) 
error_prediction_copies = np.repeat(error_prediction, df_for_training.shape[1], axis=-1)
error_y_pred = scaler.inverse_transform(error_prediction_copies)[:,0]

error_predict_period_dates = pd.date_range(list(train_dates)[-error_months], periods=error_months, freq='MS').tolist()
# Convert timestamp to date
error_dates=[]
for time_i in error_predict_period_dates:
    error_dates.append(time_i.date())
    
error_forecast = pd.DataFrame({'date':np.array(error_dates), 'open':error_y_pred})
error_forecast['date']=pd.to_datetime(error_forecast['date'])

mse = mean_squared_error(original['open'], error_forecast['open'],multioutput='raw_values')
print("MSE: "+str(mse))

sns.lineplot(x='date',y='open', data=original[['date','open']])
   
ax = sns.lineplot(x='date',y='open', data=y_pred_future[['date','open']])
plt.xticks(rotation=45)

plt.savefig('figures/'+companies[c_index]+"_"+'open'+".pdf")
plt.show()
plt.clf()


