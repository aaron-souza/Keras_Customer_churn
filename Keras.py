import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from bokeh.plotting import figure
from bokeh.io import show

#Importing data
data = pd.read_csv('D:\\Study\\Python\\Assignment2\\Keras\\Churn_Modeling.csv')
#Training and testing data
X = data.iloc[:,3:13].values 
#Target variable
Y = data.iloc[:, 13].values 

# Encoding categorical (string based) data for Countries France and Germany to 0 and 1
#print(X[:8,1], '... will now become: ')
label_X_country_encoder = LabelEncoder()
X[:,1] = label_X_country_encoder.fit_transform(X[:,1])
#print(X[:8,1])

# Encoding categorical (string based) data for Gender Male and Female to 0 and 1
#print(X[:6,2], '... will now become: ')
label_gender_encoder = LabelEncoder()
X[:,2] = label_gender_encoder.fit_transform(X[:,2])
#print(X[:6,2])

# Converting the string features into their own dimensions.
countryhotencoder = OneHotEncoder(categorical_features = [1]) # 1 is the country column
X = countryhotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training and Testing set.
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initializing the Artificial Neural Network
classifier = Sequential()

# Adding the input layer and the first hidden layer 
classifier.add(Dense(activation = 'relu', input_dim = 11, units=6, kernel_initializer='uniform'))

# Adding the hidden layer
classifier.add(Dense(activation = 'relu', units=6, kernel_initializer='uniform')) 

# Adding the hidden layer
classifier.add(Dense(activation = 'relu', units=6, kernel_initializer='uniform')) 

# Adding the output layer
classifier.add(Dense(activation = 'sigmoid', units=1, kernel_initializer='uniform')) 

#Compiling the network
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#Fit the network
values = classifier.fit(X_train, y_train, batch_size=10, epochs=50)

accuracy=values.history['acc']
#print("Accuracy: ",accuracy)

predict = classifier.predict(X_test)
predict = (predict > 0.5)
cmat = confusion_matrix(y_test, predict)
#print(cmat)
print ("Testing accuracy: ",int(((cmat[0][0]+cmat[1][1])*100)/(cmat[0][0]+cmat[1][1]+cmat[0][1]+cmat[1][0])), '%')

epocharr=[]
for i in range(0,50):
    epocharr.append(i)
    
#print("epo arrr ",epocharr)  

#Figure 1
p = figure(x_axis_label ='Epochs', y_axis_label ='Accuracy')
p.line(epocharr, accuracy, color = 'red')
p.circle(epocharr, accuracy, fill_color='yellow', size=4)
show(p)