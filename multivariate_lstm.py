

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
c_index=0
#Read the csv file
df = pd.read_csv(companies[c_index]+'.csv')

n_future = 1  # Number of months we want to look into the future based on the past months.
n_past = 3 # Number of past months we want to use to predict the future.
n_months_future = 5 #predict months in future
plot_x_count = 16 #how many dates should we show in the plot
#Separate dates for future plotting
df['date']=pd.to_datetime(df['date'], format = '%Y-%m')

all_dates = pd.to_datetime(df['date']).tolist()
date_future_first=all_dates[-n_months_future]

cols = list(df)[1:5]
print(cols) #['max', 'min', 'open', 'close']


train_df = df.copy(deep=True)
train_cond=train_df['date'] >= pd.to_datetime(date_future_first)
train_rows = train_df.loc[train_cond,:]#current
train_df.drop(train_rows.index, inplace=True)
df_for_training = train_df[cols].astype(float)



error_df = df.copy(deep=True)
error_cond=error_df['date'] < pd.to_datetime(date_future_first)
error_rows = df.loc[error_cond,:]#current
error_df.drop(error_rows.index, inplace=True)
df_for_error = error_df[cols].astype(float)



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

#prediction configs
epochs=100
batch_size=4
optimizer='sgd'
loss='mse'#mean_absolute_percentage_error mse
activation='relu'
validation_split=0.25
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
model.add(LSTM(128, activation=activation, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(64, activation=activation, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(2))
model.add(Dense(trainY.shape[2]))

#optimizer = keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer, loss=loss)
model.summary()
#scores = model.evaluate()#trainX, trainY, verbose=1)
#print("Accuracy: %.2f%%" % (scores*100))
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
prediction_future=np.array(prediction_future)
inverse_prediction = scaler.inverse_transform(prediction_future)[:]    
    
predict_period_dates = pd.date_range(date_future_first, periods=n_months_future, freq='MS').tolist()
#predict_period_dates = predict_period_dates[1:]
#prediction_future_np = np.array(prediction_future).reshape(-1, 1)
#prediction_copies = np.repeat(prediction_future_np, df_for_training.shape[1], axis=1)
#inverse_prediction = scaler.inverse_transform(prediction_copies)[:]
predicted_future = pd.DataFrame(inverse_prediction, columns=cols)

# Convert timestamp to date
forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())

date_forecast=pd.DataFrame({'date':np.array(forecast_dates)})
date_forecast['date']=pd.to_datetime(date_forecast['date'], format = '%Y-%m')#.dt.to_period('M')
predicted_future['date'] = date_forecast
#date for the graph

date_from_plot = list(all_dates)[-plot_x_count]


original = df[['date', 'max','min','open','close']]
original = df.copy()
original['date']=pd.to_datetime(original['date'], format = '%Y-%m')#.dt.to_period('M')
original = original.loc[original['date'] >= pd.to_datetime(date_from_plot, format = '%Y-%m')]#.to_period('M')]



plot_values=pd.DataFrame(original['date'], columns = ['date'])
plot_values=pd.to_datetime(plot_values['date'], format = '%Y-%m').dt.to_period('M')

#move the prediction to error 
#date_current= list(error_dates)[-1]
#cond=y_pred_future['date'] <= pd.to_datetime(date_current)
#rows = y_pred_future.loc[cond,:]#current
#y_pred_future.drop(rows.index, inplace=True)
#move predictions to errors
#frame = [error_y_pred,rows]
#error_y_pred = pd.concat(frame, ignore_index=True)


mse = []   
rms = []   
mape = []   
r2 = []
# plot the graph
for col in cols:
    #frames = [error_y_pred[['date',col]], y_pred_future[['date',col]]]     
    #predicted_values = pd.concat(frames, ignore_index = True)
    ax=sns.lineplot(x='date',y=col, data=original[['date',col]],label='real')    
    ax = sns.lineplot(x='date',y=col, data=predicted_future[['date',col]],label='predicted')
    #ax.set(xticks=plot_values)
    plt.xticks(rotation=45)
    #sns.lineplot(x='date',y=col, data=error_y_pred[['date',col]],label='predicted')   
    ax.legend()

    plt.savefig('figures/'+companies[c_index]+'_'+col+".pdf")
    plt.show()
    plt.clf()
    mse.append( mean_squared_error(df_for_error[col], predicted_future[col]))    
    rms.append(mean_squared_error(df_for_error[col], predicted_future[col], squared=False)    )
    mape.append(mean_absolute_percentage_error(df_for_error[col], predicted_future[col])*100    )
    r2 .append(r2_score(df_for_error[col],predicted_future[col]))
    



error_map={"Company":companies[c_index],"MSE":mse,"RMS":rms,"MAPE":mape,"R2":r2, 
           "n_future":[n_future],"n_past" :[n_past],"n_months_future":[n_months_future],"plot_x_count":[plot_x_count],
           "epochs":[epochs],"batch_size":[batch_size],"optimizer":[optimizer],"loss":[loss],"activation": [activation], 'validation_split':[validation_split],           
           "original_open":original['open'].values,
           "error_df['date'].values" :error_df['date'].values, 
           "error_df['open'].values" :error_df['open'].values, 
           "forecast_dates":forecast_dates,
           "predicted_future_open":predicted_future['open'].values}
edf = pd.DataFrame.from_dict(error_map, orient='index')
print(edf)
edf.to_csv('figures/'+companies[c_index]+'_data'+".csv", encoding='utf-8')       


