import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Read training, validation, and test data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize the grey codes by dividing it to the max grey value.
x_train /= 255
x_test /= 255


model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(6, kernel_size=(5,5), padding='same', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # Max pooling replacing average pooling
        #tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(16, kernel_size=(5,5)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        #tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120,activation=tf.nn.relu),
        tf.keras.layers.Dense(84,activation=tf.nn.relu),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

print(model.summary())

# Training
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=3)


# Evaluation
[Loss_value, Metrics_value] = model.evaluate(x_test, y_test)
print("Loss Value:", Loss_value)
print("Metrics Value:", Metrics_value)


# One image prediction
image_index = 100
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())