#!/usr/bin/env python
# coding: utf-8

# ### import libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn 
import stat


# ## loading data

# In[2]:


dataset=pd.read_excel("demand.xls")
dataset.head()


# In[3]:


dataset.set_index('date', inplace=True)
dataset.head()


# ## missing value

# In[4]:


dataset=dataset.fillna(method="ffill")


# ## graph data

# In[5]:


dataset.plot(figsize=(20,5))
plt.title("time series")
plt.show


# ## series to supervised

# In[6]:


def to_supervised(train,n_input,n_out):
    #falten data
    data=train
    X,y=list(),list()
    in_start=0
    for _ in range(len(data)):
        in_end=in_start+ n_input
        out_end=in_end + n_out
        if out_end<=len(data):
            x_input=data[ in_start:in_end,0]
            x_input=x_input.reshape((len(x_input)))
            X.append(x_input)
            y.append(data[in_end:out_end,0])
        in_start+=1
    return array(X), array(y)    


# ## choose optimal lag Observation usinf ACF plot

# In[7]:


pip install statsmodels


# In[8]:


import statsmodels.tsa.stattools as sts


# In[9]:


sts.adfuller(dataset.issued)


# In[10]:


from statsmodels.graphics.tsaplots import plot_acf


# In[11]:


plot_acf(dataset, lags=49)
plt.show()


# In[12]:


from numpy import array


# In[13]:


n_step=1
lags=49


# In[14]:


dataset=np.array(dataset)
dataset


# In[15]:


X,y=to_supervised(dataset,n_input=lags,n_out=n_step)


# In[16]:


X.shape,y.shape


# ## Train Test Split

# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[19]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ### Scaling

# In[20]:


from sklearn.preprocessing import MinMaxScaler


# In[21]:


scaler=MinMaxScaler(feature_range=(0,1))


# In[22]:


X_train=scaler.fit_transform(X_train)
y_train=scaler.fit_transform(y_train)
X_train.shape,y_train.shape


# In[23]:


X_train


# In[24]:


y_train


# In[25]:


X_test=scaler.fit_transform(X_test)
y_test=scaler.fit_transform(y_test)
X_test.shape,y_test.shape


# In[26]:


X_test


# ### MLP

# In[27]:


from keras.models import Sequential
from keras.layers import Dense


# ### Define model

# In[28]:


model_mlp=Sequential()
model_mlp.add(Dense(10,activation="relu",input_dim=X_train.shape[1]))
model_mlp.add(Dense(20,activation="relu"))
model_mlp.add(Dense(30,activation="relu"))
model_mlp.add(Dense(n_step,activation="selu"))
model_mlp.compile(loss='mse',optimizer="adam")


# In[29]:


model_mlp.fit(X_train,y_train,epochs=100, batch_size=16)


# ## predict test set

# In[30]:


predict_mlp=model_mlp.predict(X_test)
predict_mlp.shape


# In[31]:


predict_mlp


# In[32]:


y_test.shape


# ## MSE

# In[33]:


from sklearn.metrics import mean_squared_error


# In[34]:


mse_mlp=mean_squared_error(y_test,predict_mlp)


# In[35]:


mse_mlp


# ## Inverse_transfrom

# In[37]:


inv_y_test=scaler.inverse_transform(y_test)
inv_y_pred=scaler.inverse_transform(predict_mlp)


# In[38]:


fig=plt.figure(figsize=(15,10))
plt.plot(inv_y_test,color='b',label="Real")
plt.plot(inv_y_pred,color='r',label="predicted")


# ## predict next step

# In[39]:


type(X_test)


# In[41]:


input_samples=X_test[-1:,:]


# In[42]:


input_samples.shape


# In[43]:


next_step=model_mlp.predict(input_samples)
next_step=scaler.inverse_transform(next_step)


# In[44]:


next_step


# ## simple RNN

# ## 3d input shpe

# In[45]:


X_train_3d=X_train.reshape(X_train.shape[0],X_train.shape[1],1)                         
X_test_3d=X_test.reshape(X_test.shape[0],X_test.shape[1],1)                         


# In[46]:


X_train_3d.shape,X_test_3d.shape


# In[48]:


from keras.layers import SimpleRNN


# In[49]:


model_simple_RNN=Sequential()
model_simple_RNN.add(SimpleRNN(10,activation="relu",input_shape=(X_train_3d.shape[1],X_train_3d.shape[2]),return_sequences=True))
model_simple_RNN.add(SimpleRNN(20,activation="relu",return_sequences=True))
model_simple_RNN.add(SimpleRNN(30,activation="relu",return_sequences=False))
model_simple_RNN.add(Dense(n_step))
model_simple_RNN.compile(optimizer="adam", loss='mse')


# In[50]:


model_simple_RNN.fit(X_train_3d,y_train,epochs=100,batch_size=16)


# ## predict test

# In[51]:


predict_simpleRNN=model_simple_RNN.predict(X_test_3d)
predict_simpleRNN.shape


# In[52]:


mse_simpleRNN=mean_squared_error(y_test,predict_simpleRNN)
mse_simpleRNN


# In[53]:


inv_y_pred_simpleRnn=scaler.inverse_transform(predict_simpleRNN)


# In[54]:


fig=plt.figure(figsize=(15,10))
plt.plot(inv_y_test[:,:2],color='b',label="Real")
plt.plot(inv_y_pred_simpleRnn,color='r',label="predicted")


# In[55]:


input_samples=X_test_3d[-1:,:,:]
next_step_simpleRNN=model_simple_RNN.predict(input_samples)
next_step_simpleRNN=scaler.inverse_transform(next_step_simpleRNN)
next_step_simpleRNN


# ## GRU

# In[56]:


from keras.layers import GRU


# In[57]:


model_Gru=Sequential()
model_Gru.add(GRU(10,activation="relu",input_shape=(X_train_3d.shape[1],X_train_3d.shape[2]),return_sequences=True))
model_Gru.add(GRU(20,activation="relu",return_sequences=True))
model_Gru.add(GRU(30,activation="relu",return_sequences=False))
model_Gru.add(Dense(n_step))
model_Gru.compile(optimizer="adam", loss='mse')


# In[58]:


model_Gru.fit(X_train_3d,y_train,epochs=100,batch_size=16)


# In[59]:


predict_Gru=model_Gru.predict(X_test_3d)
predict_Gru.shape


# In[60]:


mse_Gru=mean_squared_error(y_test,predict_Gru)
mse_Gru


# In[61]:


inv_y_pred_Gru=scaler.inverse_transform(predict_Gru)


# In[62]:


fig=plt.figure(figsize=(15,10))
plt.plot(inv_y_test[:,:2],color='b',label="Real")
plt.plot(inv_y_pred_Gru,color='r',label="predicted")


# In[63]:


input_samples=X_test_3d[-1:,:,:]
next_step_Gru=model_Gru.predict(input_samples)
next_step_Gru=scaler.inverse_transform(next_step_Gru)
next_step_Gru


# ## LSTM

# In[64]:


from keras.layers import LSTM


# In[65]:


model_LSTM=Sequential()
model_LSTM.add(LSTM(10,activation="relu",input_shape=(X_train_3d.shape[1],X_train_3d.shape[2]),return_sequences=True))
model_LSTM.add(LSTM(20,activation="relu",return_sequences=True))
model_LSTM.add(LSTM(30,activation="relu",return_sequences=False))
model_LSTM.add(Dense(n_step))
model_LSTM.compile(optimizer="adam", loss='mse')


# In[66]:


model_LSTM.fit(X_train_3d,y_train,epochs=100,batch_size=16)


# In[67]:


predict_LSTM=model_LSTM.predict(X_test_3d)
predict_LSTM.shape


# In[68]:


mse_LSTM=mean_squared_error(y_test,predict_LSTM)
mse_LSTM


# In[69]:


inv_y_pred_LSTM=scaler.inverse_transform(predict_LSTM)
inv_y_test=scaler.inverse_transform(y_test)


# In[70]:


fig=plt.figure(figsize=(15,10))
plt.plot(inv_y_test,color='b',label="Real")
plt.plot(inv_y_pred_LSTM,color='r',label="predicted")


# In[71]:


input_samples=X_test_3d[-1:,:,:]
next_step_LSTM=model_LSTM.predict(input_samples)
next_step_LSTM=scaler.inverse_transform(next_step_LSTM)
next_step_LSTM


# ## CNN

# In[72]:


from keras.layers import Conv1D,MaxPooling1D,Flatten


# In[73]:


model_CNN=Sequential()
model_CNN.add(Conv1D(filters=32, kernel_size=3, activation="relu",input_shape=(X_train_3d.shape[1],X_train_3d.shape[2])))
model_CNN.add(Conv1D(filters=32, kernel_size=3, activation="relu"))
model_CNN.add(MaxPooling1D(pool_size=2))
model_CNN.add(Conv1D(filters=16, kernel_size=3, activation="relu"))
model_CNN.add(MaxPooling1D(pool_size=2))
model_CNN.add(Flatten())
model_CNN.add(Dense(n_step))
model_CNN.compile(optimizer="adam", loss='mse')
model_CNN.fit(X_train_3d,y_train,epochs=100,batch_size=16)


# In[74]:


predict_CNN=model_CNN.predict(X_test_3d)
predict_CNN.shape


# In[75]:


mse_CNN=mean_squared_error(y_test,predict_CNN)
mse_CNN


# In[76]:


inv_y_pred_CNN=scaler.inverse_transform(predict_CNN)


# In[77]:


fig=plt.figure(figsize=(15,10))
plt.plot(inv_y_test,color='b',label="Real")
plt.plot(inv_y_pred_CNN,color='r',label="predicted")
plt.legend()


# In[78]:


input_samples=X_test_3d[-1:,:,:]
next_step_CNN=model_CNN.predict(input_samples)
next_step_CNN=scaler.inverse_transform(next_step_CNN)
next_step_CNN


# ## Result comparision

# In[79]:


result={"MLP":mse_mlp, "SimpleRnn":mse_simpleRNN, "GRU":mse_Gru, "LSTM":mse_LSTM, "CNN":mse_CNN}


# In[80]:


result=pd.DataFrame.from_dict(result,orient="index")


# In[81]:


result


# In[82]:


result.plot(kind="bar",figsize=(5,2))


# In[ ]:




