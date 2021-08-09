import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization
import tensorflow as tf
import time
from tensorflow.keras.callbacks import TensorBoard
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras import regularizers

data_frame = pd.read_csv('prediction train dataset.csv')
print(data_frame)

Scale = MinMaxScaler()

X = np.array(data_frame.drop(['FG','LPG','LN','HN','Kero','Diesel','Reduced Crude'], axis='columns'))
X = Scale.fit_transform(X)

y = np.array(data_frame.drop(['API','Pour','Sulfur'],axis='columns'))
y = Scale.fit_transform(y)


dense_layers = [1,2,3,5]
layer_sizes = [3,5,10,13]

#for dense_layer in dense_layers:
#    for layer_size in layer_sizes:
#        NAME = "{}-dense_layers,{}-layer_sizes Crude Yield PREDICTION-{}".format(dense_layer,layer_size,int(time.time()))
#        tensor_board = TensorBoard(log_dir="logs/{}".format(NAME))
#        Model = Sequential()
#        Model.add(Dense(layer_size,kernel_initializer='normal', activation='relu',input_shape=(X.shape[1:])))
#        #Model.add(Dropout(0.2))
#        Model.add(BatchNormalization())

#        for l in range(dense_layer):
#            Model.add(Dense(units=layer_size, kernel_initializer='normal', activation='relu'))
#            #Model.add(Dropout(0.2))
#            Model.add(BatchNormalization())

#        Model.add(Dense(units=7,activation='linear'))

#        opt = tf.keras.optimizers.Adam(lr = 0.01, decay=1e-5)
#        Model.compile(loss = 'mse', optimizer = opt, metrics=['mae'])

#        history = Model.fit(X,y, batch_size = 10, epochs = 100, validation_split=0.2, callbacks=[tensor_board])


NAME = "real 2-dense_layers,10-layer_sizes Crude Yield PREDICTION-{}".format(int(time.time()))
tensor_board = TensorBoard(log_dir="logs/{}".format(NAME))
Model = Sequential()
Model.add(Dense(10, kernel_initializer='normal',activation='relu',input_shape=(X.shape[1:])))
Model.add(Dropout(0.2))
Model.add(BatchNormalization())

Model.add(Dense(units=10, kernel_initializer='normal', activation='relu'))
Model.add(BatchNormalization())

Model.add(Dense(units=10, kernel_initializer='normal', activation='relu'))
Model.add(BatchNormalization())

Model.add(Dense(units=7, kernel_initializer='uniform', activation='linear'))

opt = tf.keras.optimizers.Adam(lr = 0.01, decay=1e-5)
#Model.compile(loss = 'mse', optimizer = opt, metrics=['mae'])
#Model.fit(X,y, batch_size = 10, epochs = 100, validation_split=0.2, callbacks=[tensor_board])


#model_json = Model.to_json()
#with open("model1.json",'w') as json_file:
#    json_file.write(model_json)
#Model.save_weights("model_weight1.h5")
#print("Saved model to disk")

json_file = open('model1.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights('model_weight1.h5')
print('Loaded model from disk')

test_dataset = pd.read_csv('Prediction test dataset.csv')

sc = MinMaxScaler()

X_test = np.array(test_dataset.drop(['FG','LPG','LN','HN','Kero','Diesel','Reduced Crude'], axis='columns'))
X_test = sc.fit_transform(X_test)

y_test = np.array(test_dataset.drop(['API','Pour','Sulfur'],axis='columns'))
y_test = sc.fit_transform(y_test)

loaded_model.compile(loss = 'mse', optimizer = opt, metrics=['mse'])
score = loaded_model.evaluate(X_test,y_test,verbose=0)
print(score)
y_pred = loaded_model.predict(X_test)

y_pred = sc.inverse_transform(y_pred)
y_test = sc.inverse_transform(y_test)
#print(y_pred,y_test)

columns = ['FG','LPG','LN','HN','Kero','Diesel','Reduced Crude']

df_pred = pd.DataFrame(y_pred,columns=columns)
df_test = pd.DataFrame(y_test,columns=columns)

#print(df_pred)
#print(df_test)

range_list =[]
for _ in range(3):
    range_list.append(_)

#plt.ylim(0,120)
#plt.yscale(100)
#plt.plot(range_list,df_pred['Reduced Crude'],label='predicted Reduced Crude')
#plt.plot(range_list,df_test['Reduced Crude'],label='Actual Reduced Crude')
#plt.xlabel('Range of values')
#plt.ylabel("FG")
#plt.legend()
#plt.show()
















data_frame = pd.read_csv('prediction of diesel pour point train dataset.csv')
print(data_frame)

Scale = MinMaxScaler()

X = np.array(data_frame.drop(['Diesel pour point'], axis='columns'))
X = Scale.fit_transform(X)

y = np.array(data_frame.drop(['API','Pour','Sulfur'],axis='columns'))
y = Scale.fit_transform(y)

#dense_layers = [1,2,3,5]
#layer_sizes = [3,5,10,13]

#for dense_layer in dense_layers:
#    for layer_size in layer_sizes:
#        NAME = "{}-dense_layers,{}-layer_sizes Diesel pour point PREDICTION-{}".format(dense_layer,layer_size,int(time.time()))
#        tensor_board = TensorBoard(log_dir="logs/{}".format(NAME))
#        Model = Sequential()
#        Model.add(Dense(layer_size,kernel_initializer='uniform', activation='sigmoid',input_shape=(X.shape[1:])))
#        Model.add(Dropout(0.2))
#        Model.add(BatchNormalization())

#        for l in range(dense_layer):
#            Model.add(Dense(units=layer_size, kernel_initializer='uniform', activation='sigmoid'))
#            Model.add(Dropout(0.2))
#            Model.add(BatchNormalization())

#        Model.add(Dense(units=1,activation='linear'))

#        opt = tf.keras.optimizers.Adam(lr = 0.01, decay=1e-5)
#        Model.compile(loss = 'mse', optimizer = opt, metrics=['mae'])

#        history = Model.fit(X,y, batch_size = 10, epochs = 100, validation_split=0.2, callbacks=[tensor_board])


NAME = "real 2-dense_layers,13-layer_sizes Crude Yield PREDICTION-{}".format(int(time.time()))
tensor_board = TensorBoard(log_dir="logs/{}".format(NAME))
Model = Sequential()
Model.add(Dense(13, kernel_initializer='uniform',activation='sigmoid',input_shape=(X.shape[1:])))
Model.add(Dropout(0.2))
Model.add(BatchNormalization())

Model.add(Dense(units=13, kernel_initializer='uniform', activation='sigmoid'))
Model.add(Dropout(0.2))
Model.add(BatchNormalization())

Model.add(Dense(units=13, kernel_initializer='uniform', activation='sigmoid'))
Model.add(Dropout(0.2))
Model.add(BatchNormalization())

Model.add(Dense(units=1, activation='linear'))

opt = tf.keras.optimizers.Adam(lr = 0.01, decay=1e-5)
#Model.compile(loss = 'mse', optimizer = opt, metrics=['mae'])
#Model.fit(X,y, batch_size = 10, epochs = 100, validation_split=0.2, callbacks=[tensor_board])


#model_json = Model.to_json()
#with open("model2.json",'w') as json_file:
#    json_file.write(model_json)
#Model.save_weights("model_weight2.h5")
#print("Saved model to disk")

json_file = open('model2.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights('model_weight2.h5')
print('Loaded model from disk')


test_dataset = pd.read_csv('prediction of diesel pour point test dataset.csv')

sc = MinMaxScaler()

X_test = np.array(test_dataset.drop(['Diesel pour point'], axis='columns'))
X_test = sc.fit_transform(X_test)

y_test = np.array(test_dataset.drop(['API','Pour','Sulfur'],axis='columns'))
y_test = sc.fit_transform(y_test)

loaded_model.compile(loss = 'mse', optimizer = opt, metrics=['mse'])
score = loaded_model.evaluate(X_test,y_test,verbose=0)
print(score)
y_pred = loaded_model.predict(X_test)

y_pred = sc.inverse_transform(y_pred)
y_test = sc.inverse_transform(y_test)
#print(y_pred)
#print(y_test)

range_list =[]
for _ in range(3):
    range_list.append(_)

#plt.ylim(0,100)
#plt.plot(range_list,y_pred,label='predicted Diesel Pour Point')
#plt.plot(range_list,y_test,label='Actual Diesel Pour Point')
#plt.xlabel('Range of values')
#plt.ylabel("Diesel Pour Point")
#plt.legend()
#plt.show()










data_frame = pd.read_csv('Prediction of Hydrocracker Total Gasoline Yield Training dataset.csv')
print(data_frame)

Scale = MinMaxScaler()

X = np.array(data_frame.drop(['Total gasoline yield'], axis='columns'))
X = Scale.fit_transform(X)

y = np.array(data_frame.drop(['API','K','SCFB H2'],axis='columns'))
y = Scale.fit_transform(y)

dense_layers = [1,2,3,5]
layer_sizes = [3,5,10,13]
#for dense_layer in dense_layers:
#    for layer_size in layer_sizes:
#        NAME = "{}-dense_layers,{}-layer_sizes Hydrocracker Total Gasoline Yield PREDICTION-{}".format(dense_layer,layer_size,int(time.time()))
#        tensor_board = TensorBoard(log_dir="logs/{}".format(NAME))
#        Model = Sequential()
#        Model.add(Dense(layer_size,kernel_initializer='normal', activation='relu',input_shape=(X.shape[1:])))
#        Model.add(Dropout(0.2))
#        Model.add(BatchNormalization())

#        for l in range(dense_layer):
#            Model.add(Dense(units=layer_size, kernel_initializer='normal', activation='relu'))
#            Model.add(Dropout(0.2))
#            Model.add(BatchNormalization())

#        Model.add(Dense(units=1,activation='linear'))

#        opt = tf.keras.optimizers.Adam(lr = 0.01, decay=1e-5)
#        Model.compile(loss = 'mse', optimizer = opt, metrics=['mae'])

#        history = Model.fit(X,y, batch_size = 10, epochs = 100, validation_split=0.2, callbacks=[tensor_board])


NAME = "real 2-dense_layers,5-layer_sizes Hydrocracker Total Gasoline Yield PREDICTION-{}".format(int(time.time()))
tensor_board = TensorBoard(log_dir="logs/{}".format(NAME))
Model = Sequential()
Model.add(Dense(5, kernel_initializer='normal',activation='relu',input_shape=(X.shape[1:])))
Model.add(Dropout(0.2))
Model.add(BatchNormalization())

Model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))
Model.add(Dropout(0.2))
Model.add(BatchNormalization())

Model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))
Model.add(Dropout(0.2))
Model.add(BatchNormalization())

Model.add(Dense(units=1, activation='linear'))

opt = tf.keras.optimizers.Adam(lr = 0.01, decay=1e-5)
#Model.compile(loss = 'mse', optimizer = opt, metrics=['mae'])
#Model.fit(X,y, batch_size = 10, epochs = 100, validation_split=0.2, callbacks=[tensor_board])


#model_json = Model.to_json()
#with open("model3.json",'w') as json_file:
#    json_file.write(model_json)
#Model.save_weights("model_weight3.h5")
#print("Saved model to disk")

json_file = open('model3.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights('model_weight3.h5')
print('Loaded model from disk')

test_dataset = pd.read_csv('Prediction of Hydrocracker Total Gasoline Yield Training dataset.csv')

sc = MinMaxScaler()

X_test = np.array(test_dataset.drop(['Total gasoline yield'], axis='columns'))
X_test = sc.fit_transform(X_test)

y_test = np.array(test_dataset.drop(['API','K','SCFB H2'],axis='columns'))
y_test = sc.fit_transform(y_test)

loaded_model.compile(loss = 'mse', optimizer = opt, metrics=['mse'])
score = loaded_model.evaluate(X_test,y_test,verbose=0)
print(score)
y_pred = loaded_model.predict(X_test)

y_pred = sc.inverse_transform(y_pred)
y_test = sc.inverse_transform(y_test)
print(y_pred)
print(y_test)

range_list =[]
for _ in range(len(y_test)):
    range_list.append(_)

plt.ylim(0,200)
plt.plot(range_list,y_pred,label='predicted Total gasoline yield')
plt.plot(range_list,y_test,label='Actual Total gasoline yield')
plt.xlabel('Range of values')
plt.ylabel("Total gasoline yield")
plt.legend()
plt.show()