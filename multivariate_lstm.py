

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
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

#from datetime import datetime
companies = ['Комерцијална банка Скопје', 'Алкалоид Скопје','Гранит Скопје','Макпетрол Скопје','Македонијатурист Скопје']
c_index=4
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
print(train_dates.tail(18)) #Check last few dates.

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


#prediction configs
epochs=100
batch_size=4
optimizer='sgd'
loss='mse'
activation='relu'
validation_split=0.2
#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (180, 5)
for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future,:])


trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

# define the Autoencoder model

model = Sequential()
model.add(LSTM(64, activation=activation, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation=activation, return_sequences=False))


model.add(Dropout(0.1))
model.add(Dense(2))
model.add(Dense(trainY.shape[2]))

model.compile(optimizer=optimizer, loss=loss)
model.summary()

# fit the model
history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1,  shuffle= False)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
 
plt.savefig('figures/'+companies[c_index]+"_loss.pdf")
plt.show()
plt.clf()

#Remember that we can only predict one month in future as our model needs 5 variables
#as inputs for prediction. We only have all 5 variables until the last month in our dataset.
# generate the multi-step forecasts
prediction_future = []

x_pred = trainX[-1:,:,:]  # last observed input sequence
y_pred = trainY[-1]         # last observed target value
#Make prediction
#for i in range(n_months_for_prediction):
    # feed the last forecast back to the model as an input

#predict months in future
n_months_future = 6
#error for past months
error_months = 6

for i in range(n_months_future):
    a=x_pred[:, 1:, :]    
    b= y_pred.reshape(1,1,df_for_training.shape[1])
    #b = np.repeat(b, df_for_training.shape[1], axis=-1)

    x_pred = np.append(a, b, axis=1)
    # generate the next forecast
    y_pred = model.predict(x_pred) 
    flat = y_pred.flatten()
    # save the forecast
    prediction_future.append(flat)
    

#Perform inverse transformation to rescale back to original range
#Since we used 5 variables for transform, the inverse expects same dimensions
#Therefore, let us copy our values 5 times and discard them after inverse transform
    # transform the forecasts back to the original scale
inverse_prediction = scaler.inverse_transform(prediction_future)[:]    
    
predict_period_dates = pd.date_range(list(train_dates)[-1], periods=n_months_future, freq='MS').tolist()
#prediction_future_np = np.array(prediction_future).reshape(-1, 1)
#prediction_copies = np.repeat(prediction_future_np, df_for_training.shape[1], axis=1)
#inverse_prediction = scaler.inverse_transform(prediction_copies)[:]
y_pred_future = pd.DataFrame(inverse_prediction, columns=cols)

# Convert timestamp to date
forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())

df_forecast=pd.DataFrame({'date':np.array(forecast_dates)})
df_forecast['date']=pd.to_datetime(df_forecast['date'], format = '%Y-%m-%d')
y_pred_future['date'] = df_forecast

#date for the graph
date_from_error = list(train_dates)[-(error_months)]#add one yaer on the error months
date_from_plot = list(train_dates)[-(error_months+12)]


original = df[['date', 'max','min','open','close']]
original = df.copy()
original['date']=pd.to_datetime(original['date'])
original = original.loc[original['date'] >= date_from_plot]

# calculate the errors
error_original = original.loc[original['date'] >= date_from_error]

error_suspect=trainY[-error_months:] 
#error_prediction_copies = np.repeat(error_prediction, df_for_training.shape[1], axis=-1)
error_y_pred = scaler.inverse_transform(error_suspect)

error_predict_period_dates = pd.date_range(list(train_dates)[-error_months], periods=error_months, freq='MS').tolist()
# Convert timestamp to date
error_dates=[]
for time_i in error_predict_period_dates:
    error_dates.append(time_i.date())
error_y_pred['date']=np.array(error_dates)
#error_forecast = pd.DataFrame({'date':np.array(error_dates), 'max':error_y_pred[0], 'min':error_y_pred[1], 'open':error_y_pred[2], 'close':error_y_pred[3]})
#error_forecast['date']=pd.to_datetime(error_forecast['date'])

mse = []   
rms = []   
mape = []   
r2 = []
# plot the graph
for col in cols:
    sns.lineplot(x='date',y=col, data=original[['date',col]],label='real')   
    sns.lineplot(x='date',y=col, data=error_y_pred[['date',col]],label='predicted current')   
    ax = sns.lineplot(x='date',y=col, data=y_pred_future[['date',col]],label='predicted future')
    ax.legend()
    plt.xticks(rotation=45)
    plt.savefig('figures/'+companies[c_index]+'_'+col+".pdf")
    plt.show()
    plt.clf()
    mse.append( mean_squared_error(error_original[col], error_y_pred[col]))    
    rms.append(mean_squared_error(error_original[col], error_y_pred[col], squared=False)    )
    mape.append(mean_absolute_percentage_error(error_original[col], error_y_pred[col])*100    )
    r2 .append(r2_score(error_original[col],error_y_pred[col]))
    



error_map={"Company":companies[c_index],"MSE":[mse],"RMS":[rms],"MAPE":[mape],"R2":[r2], 
           "n_future":[n_future],"n_past" :[n_past],"n_months_future":[n_months_future], 
           "epochs":[epochs],"batch_size":[batch_size],"optimizer":[optimizer],"loss":[loss],"activation": [activation], 'validation_split':[validation_split],
           "error_months":[error_months],"date_from_error" :[date_from_error],"date_from_plot" :[date_from_plot],           
           "predictions_dates":error_forecast['date'].values,"original_open":original['open'].values,"error_y_pred_open":error_y_pred['open'].values}
edf = pd.DataFrame.from_dict(error_map, orient='index')
edf.to_csv('figures/'+companies[c_index]+'_data'+".csv", encoding='utf-8', index=False)       


