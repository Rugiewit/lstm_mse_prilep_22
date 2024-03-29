

"""
dataset: macedonian exchange office
based on  https://youtu.be/tepxdcepTbY
"""
import io
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import pingouin as pg
from scipy import stats
from tensorflow.python.client import device_lib

def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

print(device_lib.list_local_devices())
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

companies = ['Комерцијална банка Скопје', 'Алкалоид Скопје','Гранит Скопје','Макпетрол Скопје','Македонијатурист Скопје']
companies_short_names= ['KMB','ALK','GRNT','MPT','MTUR']
c_index=0
n_months_future = 6 #predict months in future
#Read the csv file
df = pd.read_csv(companies[c_index]+'.csv')

n_future = 1  # Number of months we want to look into the future based on the past months.
n_past = 3 # Number of past months we want to use to predict the future.

plot_x_count = 20 #how many dates should we show in the plot
#Separate dates for future plotting
df['date']=pd.to_datetime(df['date'], format = '%Y-%m')
df.describe()
all_dates = pd.to_datetime(df['date']).tolist()
date_future_first=all_dates[-1]

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
df_for_error_scaled = scaler.transform(df_for_error)

#As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
#In this example, the n_features is 5. We will make timesteps = 14 (past days data used for training).

#Empty lists to be populated using formatted training data
trainX = []
trainY = []
#Empty lists to be populated using formatted training data
errorX = []
errorY = []

#prediction configs
epochs=100
batch_size=4
optimizer='sgd' #sgd # RMSprop #adam
loss='mse'#mean_absolute_percentage_error mse
activation='relu' #relu # tanh  #
validation_split=0.3
#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (180, 5)
for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future,:])

for i in range(n_past, len(df_for_error_scaled) - n_future +1):
    errorX.append(df_for_error_scaled[i - n_past:i, 0:df_for_error.shape[1]])
    errorY.append(df_for_error_scaled[i + n_future - 1:i + n_future,:])


trainX, trainY = np.array(trainX), np.array(trainY)
errorX, errorY = np.array(errorX), np.array(errorY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

# define the Autoencoder model
model = Sequential()
model.add(LSTM(128, activation=activation, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation=activation, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(16))
model.add(Dense(trainY.shape[2]))

#optimizer = keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

scores = model.evaluate(errorX,errorY,verbose=0)#trainX, trainY, verbos
scores_loss=0
scores_acc=0
if(len(scores)>1):
    scores_loss=scores[0]
    scores_acc=scores[1]*100
    print("loss: %f" % (scores_loss))
    print("Accuracy: %f" % (scores_acc))

model.summary()


# fit the model
history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=1,  shuffle= False)


plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.text(80, 0.8,companies_short_names[c_index],weight='bold')

plt.legend()
 
plt.savefig('figures/'+companies[c_index]+"_"+str(n_months_future)+"_loss.eps",format='eps')
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

mae = []
mse = []   
rms = []   
mape = []   
r2 = []
x=[]
y=[]
statistics=[]
# plot the graph
for col in cols:
    #frames = [error_y_pred[['date',col]], y_pred_future[['date',col]]]     
    #predicted_values = pd.concat(frames, ignore_index = True)
    ax=sns.lineplot(x='date',y=col, data=original[['date',col]],label='real')  
    if(len(forecast_dates)==1):
        ax = sns.scatterplot(x='date',y=col, data=predicted_future[['date',col]],label='predicted')
    else:
        ax = sns.lineplot(x='date',y=col, data=predicted_future[['date',col]],label='predicted')
    #ax.set(xticks=plot_values)
    plt.xticks(rotation=75)
    #sns.lineplot(x='date',y=col, data=error_y_pred[['date',col]],label='predicted')   
    ax.legend()
    plt.text(0.2, 0.6,companies_short_names[c_index],horizontalalignment='center',verticalalignment='center',weight='bold',transform = ax.transAxes)

    plt.savefig('figures/'+companies[c_index]+'_'+col+"_"+str(n_months_future)+".eps",format='eps')
    plt.show()
    plt.clf()
    #errors
    x.extend(df_for_error[col])
    y.extend(predicted_future[col])
    
x=np.asarray(x)
y=np.asarray(y)
mae.append( mean_absolute_error(x, y))    
mse.append( mean_squared_error(x, y))    
rms.append(mean_squared_error(x, y, squared=False)    )
mape.append(mean_absolute_percentage_error(x, y)*100    )
r2.append(np.corrcoef(x, y)[0,1]**2)
#regression    
# Using a NumPy array:
#################################
# real is depndent variable
# correlation real data and predicted vales
# SWAP x and Y

slope, intercept, r_value, p_value, std_err = stats.linregress(y,x)
analysis_df=pg.linear_regression(y,x)
print(analysis_df)
statistics.append("slope: "+str(slope))
statistics.append("intercept: "+str(intercept))
statistics.append("r_value(correlation): "+str(r_value))
statistics.append("r^2_value(determination): "+str(r_value**2)) 
statistics.append("p_value: "+str(p_value))
statistics.append("std_err: "+str(std_err))
statistics.append("a_r2: "+str('%.3f' % analysis_df['r2'][1])) 
statistics.append("b_r2: "+str('%.3f' % analysis_df['r2'][0])) 
statistics.append("a_t: "+str('%.3f' % analysis_df['T'][1])) 
statistics.append("b_t: "+str('%.3f' % analysis_df['T'][0])) 
statistics.append("a_p_value: "+str('%.3f' % analysis_df['pval'][1]))
statistics.append("b_p_value: "+str('%.3f' % analysis_df['pval'][0]))
statistics.append("a_std_err: "+str('%.3f' % analysis_df['se'][1]))
statistics.append("b_std_err: "+str('%.3f' % analysis_df['se'][0]))

print(statistics)
plt.scatter(y, x, color = "b", marker = "o", s = 30)   
# predicted response vector
y_pred = intercept + slope*y  
# plotting the regression line
plt.plot(y, y_pred, color = "g")  
plt.text(0.8, 0.1,'y='+('%.2f' % slope)+'*x+'+('%.2f' % intercept), bbox=dict(facecolor='white'),horizontalalignment='center',verticalalignment='center',weight='bold', transform=ax.transAxes)
# putting labels
plt.text(0.1, 0.9,companies_short_names[c_index],horizontalalignment='center',verticalalignment='center',weight='bold', transform=ax.transAxes)

plt.xlabel('Prediction')
plt.ylabel('Real')  
plt.savefig('figures/'+companies[c_index]+'_regression_'+str(n_months_future)+".eps",format='eps')
###################################################
# function to show plot
plt.show()
plt.clf()
    

summary=get_model_summary(model)
accuracy = str(history.history['accuracy'][-1])
val_loss= str(history.history['val_loss'][-1])
model_loss= str(history.history['loss'][-1])
error_map={"Company":companies[c_index],"MAE":mae,"MSE":mse,"RMS":rms,"MAPE":mape,"R2":r2, "statistics":statistics,
           "analysis_df":analysis_df,
           "n_future":[n_future],"n_past" :[n_past],"n_months_future":[n_months_future],"plot_x_count":[plot_x_count],
           "epochs":[epochs],"batch_size":[batch_size],"optimizer":[optimizer],"loss_alg":[loss],"activation": [activation],'validation_split':[validation_split],  
           "loss":  scores_loss,
           "accuracy_model": scores_acc*100,
           "original_open":original['open'].values,
           "error_df['date'].values" :error_df['date'].values, 
           "error_df['open'].values" :error_df['open'].values, 
           "predicted_future['date']":predicted_future['date'],
           "predicted_future_open":predicted_future['open'].values,
           "df.describe":df.describe(),
           "model_loss:":model_loss,
           "accuracy:":accuracy,
           "val_loss:":val_loss,
           "model.summary":summary}

edf = pd.DataFrame.from_dict(error_map, orient='index')

edf.to_csv('figures/'+companies[c_index]+"_"+str(n_months_future)+'_data'+".csv", encoding='utf-8')            

print('Ta=%.3f' % analysis_df['T'][1])
print('Pa%.3f' % analysis_df['pval'][1])
print('SEa%.3f' % analysis_df['se'][1])

