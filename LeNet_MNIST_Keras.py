import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Import the required Keras modules containing model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


#Read training and test data from MNIST dataset 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

'''
image_index = 100
print(y_train[image_index]) 
plt.imshow(x_train[image_index], cmap='Greys')
'''

# Reshape the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize the grey codes by dividing it to the max grey value.
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])


# Create a Sequential Model and adding the layers
model = Sequential()
#First Convolutional Layer
model.add(Conv2D(6, kernel_size=(5,5), padding='same', input_shape=input_shape))
#First Pooling Layer
model.add(MaxPooling2D(pool_size=(2, 2))) # Max pooling is used here, instead of avg pooling
#Second Convolutional Layer
model.add(Conv2D(16, kernel_size=(5,5), input_shape=input_shape))
#Second Pooling Layer
model.add(MaxPooling2D(pool_size=(2, 2))) # Max pooling is used here, instead of avg pooling
#First Fully Connected Layer
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
#Second Fully Connected Layer
#model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

print(model.summary())

# Training
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs=3)

# Evaluation
[Loss_value, Metrics_value] = model.evaluate(x_test, y_test)
print("Loss Value:", Loss_value)
print("Metrics Value:", Metrics_value)


# One image prediction
image_index = 100
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())
