import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

#from tensorflow.keras import optimizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size    = 128
epochs        = 200
iterations    = 391
num_classes   = 10
mean          = [125.307, 122.95, 113.865]
std           = [62.9932, 62.0887, 66.7048]

def build_model():
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', input_shape=(32,32,3)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal'))
    # sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def scheduler(epoch):
    if epoch < 100:
        return 0.01
    if epoch < 150:
        return 0.005
    return 0.001

if __name__ == '__main__':

    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    
    fig = plt.figure(figsize=(8,3))

    num_classes = 10
    for i in range(num_classes):
        ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
        idx = np.where(y_train[:]==i)[0]
        features_idx = x_train[idx,::]
        img_num = np.random.randint(features_idx.shape[0])
        #im = np.transpose(features_idx[img_num,::], (1, 2, 0))
        im = features_idx[img_num,::]
        ax.set_title(class_names[i])
        plt.imshow(im)
    
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # data preprocessing  [raw - mean / std]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
        
    # build network
    model = build_model()
    print(model.summary())
    # set callback
    tb_cb = TensorBoard(log_dir='./LeNet_dp_da_CIFAR10_graph', histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr,tb_cb]

    # using real-time data augmentation
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,
            width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

    datagen.fit(x_train)

    # start train 
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=iterations,
                        epochs=epochs,
                        callbacks=cbks,
                        validation_data=(x_test, y_test))
    # save model
    model.save('LeNet_dp_da_CIFAR10.h5')
    
    # Evaluation with one image prediction
    image_index = 100
    pred = model.predict(x_test[image_index,::].reshape(1, 32, 32, 3))
        
    #im = np.transpose(x_test[image_index,::], (1, 2, 0))
    im = x_test[image_index,::]
    
    fig1 = plt.figure(figsize=(16, 8))
    fig1.suptitle(class_names[np.argmax(pred)], fontsize=16)
    plt.imshow(im)    
