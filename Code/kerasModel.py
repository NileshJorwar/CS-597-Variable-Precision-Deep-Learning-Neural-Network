
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# In[2]:


x_train.shape


# In[3]:


# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float16')
x_test = x_test.astype('float16')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])


# In[39]:


import numpy
numpy.count_nonzero(x_train[59999][27]==1)
#x_train[0].shape


# In[4]:


# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.sigmoid))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.sigmoid))


# In[9]:


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=30,batch_size=10)


# In[8]:


#Test Loss and Accuracy
model.evaluate(x_test, y_test)
#[0.05581904458472854, 0.9832] 10 epoc time 761 secs


# In[26]:


#Model split 1/3
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
y_pred = model.predict_classes(x_test)
x_pred = model.predict_classes(x_train)
print(confusion_matrix(y_test, y_pred))


# In[1]:


x_train[0][0]

