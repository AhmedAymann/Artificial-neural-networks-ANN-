#importing libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing data 

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values 
y = dataset.iloc[:, 13].values


# encoding the independant variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
ct = ColumnTransformer(
    [('onehotencoder' , OneHotEncoder(categories = 'auto'), [1])],
    remainder = 'passthrough'
    )
x = ct.fit_transform(x)

#avoid dummy variable trap
x = x[:, 1:]


#splitting data into trainingset and testset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size= 0.2, random_state= 0)




# Feature scalling  
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# importing keres library 
import keras
from keras.models import Sequential
from keras.layers import Dense

# initializing the ANN
classifier = Sequential() 

# adding the input layer and the first hidden layer
classifier.add(Dense(units= 6, kernel_initializer="uniform", activation = "relu", input_dim = 11))

# adding the Second hidden layer
classifier.add(Dense(units = 6,  kernel_initializer="uniform", activation = "relu"))

# add the output layer
classifier.add(Dense(units = 1,  kernel_initializer="uniform", activation = "sigmoid"))

# compiling the ANN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=["accuracy"])

# fitting the ANN to the training set
classifier.fit(x_train, y_train, batch_size= 10, epochs= 100)

# making the predictions and evaluating the model


# predicting the testset results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

 
    
