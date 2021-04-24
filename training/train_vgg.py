# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import MaxPool2D, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras import Sequential
import tensorflow.keras as keras
from keras.initializers import GlorotNormal, HeNormal
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.experimental.preprocessing import RandomCrop
from keras.layers import Flatten, Dropout, Dense, BatchNormalization, Activation
import matplotlib.pyplot as plt 
import pandas
from sklearn import preprocessing
from keras.utils import to_categorical
import numpy as np

input_shape = (48,48,1) 
num_classes = 7
batch_size = 128
nb_train_samples= 28709
img_height=48
img_width=48

FTRAIN= '/data/train_fer.csv'

def get_model():
  initializer = GlorotNormal()
  model= Sequential()
  # model.add(RandomCrop(48,48))
  model.add(Conv2D(filters = 64, kernel_size = (3, 3), input_shape = input_shape, padding = 'same', strides=1, kernel_initializer=initializer))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(filters = 64, kernel_size = (3, 3), input_shape = input_shape, padding = 'same', strides=1, kernel_initializer=initializer))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  # model.add(Dropout(0.5))
  model.add(MaxPool2D(pool_size=(2,2), strides=2))
  model.add(Dropout(0.6))
  model.add(Conv2D(128, (3,3),  padding ='same', strides=1, kernel_initializer=initializer))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(128, (3,3),  padding ='same', strides=1, kernel_initializer=initializer))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2,2), strides=2))
  model.add(Dropout(0.6))
  model.add(Conv2D(256, (3,3),  padding ='same', strides=1, kernel_initializer=initializer))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(256, (3,3),  padding ='same', strides=1, kernel_initializer=initializer))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2,2), strides=2))
  model.add(Dropout(0.6))
  model.add(Conv2D(512, (3,3),  padding ='same', strides=1, kernel_initializer=initializer))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(512, (3,3),  padding ='same', strides=1, kernel_initializer=initializer))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2,2), strides=2))
  model.add(Dropout(0.6))
  model.add(Flatten())
  model.add(Dense(units= 1024))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  # model.add(Dropout(0.7))
  model.add(Dense(units= num_classes, activation= 'softmax'))
  opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, decay = 0.0001)
  # cce = tf.keras.losses.CategoricalCrossentropy()
  history= model.compile(loss= 'categorical_crossentropy', optimizer= opt, metrics=['accuracy'] )
  return model
def display(history):
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

def load_data(mode):
  df = pandas.read_csv(FTRAIN)
  df['pixels']= df['pixels'].apply(lambda im: np.fromstring(im, sep=' '))
  X= np.vstack(df['pixels'])
  X_scaled = preprocessing.scale(X)
  y = df[df.columns[:-1]].values
  y_truth = y.astype(np.float32)
  if(mode == 'TRAIN'):
    X_scaled = X_scaled[:25500]
    y_truth = y_truth[:25500]
  if(mode == 'TEST'):
    X_scaled = X_scaled[25500:]
    y_truth = y_truth[25500:]
  return X_scaled, y_truth
def load2d(mode):
    if(mode == 'TEST'):
      X,y= load_data('TEST') #load as 2D array
    else:
      X,y = load_data('TRAIN')
    X = X.reshape(-1, 48, 48,1) # convert to list of samples as 2D arrays
    y= to_categorical(y)
    return X,y

class LearningRateScheduler(keras.callbacks.Callback):
    """Halve the learning rate if validation accuracy does not improve for 10 epochs.

  Arguments:
      patience: Number of epochs to wait after max has been hit(10) After this
      number of no improvement, lr is halved.
  """

    def __init__(self, patience=10):
        super(LearningRateScheduler, self).__init__()
        self.patience = patience

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when accuracy has not improved.
        self.wait = 0
        # Initialize the past value as 0.
        self.past= 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_accuracy")
        if np.greater(current, self.past): #if the current is greater than the last value
            self.past = current
            self.wait = 0
        else:
            self.wait += 1 #add the number of epochs we waited before seeing no improvement
            if self.wait >= self.patience: #if number of epochs we waited is greater than 10
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.model.optimizer.learning_rate /2)
                print("Learning rate is halved to value")
                self.wait=0

    def on_train_end(self, logs=None):
        print("Final learning rate we were using is: ");

X_train, y_train = load2d('TRAIN')
X_test, y_test = load2d('TEST')
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


model= get_model()

train_datagen = ImageDataGenerator(horizontal_flip=True,
                                   height_shift_range= 0.2, 
                                   width_shift_range= 0.2,
                                   validation_split=0.2)

# train_generator= datagen.flow(X_train,y_train)
train_generator = train_datagen.flow(
    X_train,y_train,
    batch_size=batch_size,
    subset='training') # set as training data

validation_generator = train_datagen.flow(
    X_train,y_train, # same directory as training data
    batch_size=batch_size,
    subset='validation')

history=  model.fit(train_generator, validation_data = validation_generator,  epochs= 120,  shuffle = False, verbose=1, callbacks=[LearningRateScheduler()] )

model.save('vgg.h5')
display(history)
