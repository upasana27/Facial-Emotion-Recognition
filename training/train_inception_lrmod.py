#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D,MaxPool2D, AveragePooling2D,  Flatten, Dense, Flatten, Dropout, Dense, BatchNormalization, Activation
from tensorflow.keras import Model, Sequential
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
import pandas
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
import numpy as np


input_shape = (48,48,1) 
num_classes = 7
num_filter3 = 32
initializer = GlorotNormal()
batch_size = 128
img_height=48
img_width=48
FTRAIN= 'train_fer.csv'

def inception_module(layer_in, n):
    # 1x1 conv
    conv1 = Conv2D(int(3*n/4), (1,1), padding='same')(layer_in)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    # 3x3 conv
    conv3_reduce = Conv2D(int(n/2), (1,1), padding='same')(layer_in)
    conv3_reduce = BatchNormalization()(conv3_reduce)
    conv3_reduce = Activation('relu')(conv3_reduce)
    conv3 = Conv2D(n, (3,3), padding='same')(conv3_reduce)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    # 5x5 conv
    conv5_reduce = Conv2D(int(n/2), (1,1), padding='same')(layer_in)
    conv5_reduce = BatchNormalization()(conv5_reduce)
    conv5_reduce = Activation('relu')(conv5_reduce)
    conv5 = Conv2D(int(n/4), (5,5), padding='same')(conv5_reduce)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    # pooling projection
    pool = MaxPool2D((3,3), strides=(1,1), padding='same')(layer_in)
    # pool = Dropout(0.6)(pool)
    pool = Conv2D(int(n/4), (1,1), padding='same')(pool)
    pool = BatchNormalization()(pool)
    pool = Activation('relu')(pool)
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out
def get_model():
    visible = Input(shape=input_shape)
    # visible = RandomCrop(48,48)(visible)
    layer = Conv2D(64, (3,3), padding='same', kernel_initializer=initializer)(visible)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = inception_module(layer, num_filter3)
    layer = MaxPool2D((3,3), strides=2, padding='same')(layer)
    # layer = Dropout(0.6)(layer)
    layer = inception_module(layer, num_filter3 + 32)
    layer = inception_module(layer, num_filter3 + 32 +32)
    layer = MaxPool2D((3,3), strides=2, padding='same')(layer)
    # layer = Dropout(0.6)(layer)
    layer = inception_module(layer, num_filter3 + 32 + 32 +32)
    layer = inception_module(layer, num_filter3 + 32 +32 +32 +32)
    layer = MaxPool2D((3,3), strides=2, padding='same')(layer)
    # layer = Dropout(0.6)(layer)
    layer = inception_module(layer, num_filter3 + 32 + 32 +32 + 32 +32 )
    layer = inception_module(layer, num_filter3 + 32 + 32 +32 + 32 +32 +32)
    layer = AveragePooling2D((6,6), strides=1, padding='valid')(layer)
    layer = Dropout(0.6)(layer)
    layer = Flatten()(layer)
    layer = Dense(1024)(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.6)(layer)
    layer = Dense(7, activation="softmax")(layer)
    model = Model(inputs=visible, outputs=layer)
    opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, decay = 0.0001)
    history= model.compile(loss= 'categorical_crossentropy', optimizer= opt, metrics=['accuracy'] )
    model.summary()
    return model

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


X_train, y_train = load2d('TRAIN')
X_test, y_test = load2d('TEST')
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
train_datagen = ImageDataGenerator(horizontal_flip=True,
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


model= get_model()

history=  model.fit(train_generator, validation_data = validation_generator,  epochs=120,  shuffle = False, verbose=1, callbacks = [LearningRateScheduler()] )
model.save('inception_lrmod
.h5')
display(history)





