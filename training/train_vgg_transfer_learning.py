#!/usr/bin/env python
# coding: utf-8

# In[1]:


FTRAIN= 'train_fer.csv'
import pandas
import tensorflow
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
import numpy as np
import pylab as pl
from PIL import Image
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
    X = X.reshape(-1, 48, 48)
    # print(X.shape)
    images = []
    for i in X:
      # print(i.shape)
      img = Image.fromarray(i)
      resize_img = img.resize((224,224))
      resized_matrix = np.asarray(resize_img)
      resized_matrix = np.dstack([resized_matrix, resized_matrix, resized_matrix])
      images.append(resized_matrix)
    print('X_data shape:', np.array(images).shape)
    y= to_categorical(y)
    return np.array(images),y


X_train, y_train = load2d('TRAIN')
X_test, y_test = load2d('TEST')
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

  


# In[2]:


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D,MaxPool2D, GlobalAveragePooling2D,AveragePooling2D,  Flatten, Dense, Flatten, Dropout, Dense, BatchNormalization, Activation
from tensorflow.keras import Model, Sequential
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import VGG16
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
import pandas
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
import numpy as np

def build_advanced_net(model_weights=None, image_size: int = 224, classes: int = 7) -> Sequential:

    conv_base = VGG16(include_top=False,
                      weights='imagenet',
                      input_shape=(image_size, image_size, 3))
    for layer in conv_base.layers[:-2]:
        layer.trainable = False
    model = Sequential()
    model.add(conv_base)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    if model_weights is not None:
        model.load_weights(model_weights)

    return model


# In[3]:


def test_accuracy():
  score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
  print('Test score:', score)
  print('Test accuracy:', acc)
  #code for testing and then creating confusion matrix
  pred = model.predict(X_test) #should create the predicted vector, Y_test is the true label vector
  labels_pred = np.argmax(pred,axis=1)
  true_labels = np.argmax(y_test, axis=1)
  matrix = multilabel_confusion_matrix(true_labels,labels_pred, labels=[6,5,4,3,2,1,0])
  print('Confusion matrix : \n',matrix)
  # classification report for precision, recall f1-score and accuracy
  # matrix = classification_report(true_labels,labels_pred, labels=[6,5,4,3,2,1,0])
  print('Classification report : \n', classification_report(true_labels,labels_pred, labels=[6,5,4,3,2,1,0]))


# ## batch_size = 32
# img_height= 224
# img_width= 224
# train_datagen = ImageDataGenerator(horizontal_flip=True,
#                                    validation_split=0.2)
# 
# # train_generator= datagen.flow(X_train,y_train)
# train_generator = train_datagen.flow(
#     X_train,y_train,
#     batch_size=batch_size,
#     subset='training') # set as training data
# 
# validation_generator = train_datagen.flow(
#     X_train,y_train, # same directory as training data
#     batch_size=batch_size,
#     subset='validation')
# 
# 
# # create and compile model
# print('[INFO] creating model...')
# model = build_advanced_net()
# model.summary()
# 
# opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, decay = 0.0001)
# model.compile(loss= 'categorical_crossentropy', optimizer= opt, metrics=['accuracy'] )
# print('[INFO] model compiled.')
# 
# 
# history=  model.fit(train_generator, validation_data = validation_generator,  epochs=25,  shuffle = False, verbose=1 )
# display(history)
# 

# In[10]:


batch_size = 64

img_height= 224 
img_width= 224 
train_datagen = ImageDataGenerator(horizontal_flip=True, validation_split=0.2)
train_generator = train_datagen.flow( X_train,y_train, batch_size=batch_size, subset='training') # set as training data
validation_generator = train_datagen.flow( X_train,y_train,  batch_size=batch_size, subset='validation')
model = build_advanced_net() 
opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, decay = 0.0001) 
model.compile(loss= 'categorical_crossentropy', optimizer= opt, metrics=['accuracy'] ) 
history= model.fit(train_generator, validation_data = validation_generator, epochs=25, shuffle = False, verbose=1 ) 
display(history)
model.save('vgg_transfer_learning.h5')
test_accuracy()


# In[ ]:




