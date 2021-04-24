# -*- coding: utf-8 -*-

from tensorflow import keras
from keras.models import load_model
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
import pandas
from sklearn import preprocessing
from keras.utils import to_categorical
import numpy as np
from tensorflow.keras import Model
from sklearn.metrics import accuracy_score

FTRAIN= '/data/train_fer.csv'
int num_of_models; #number of models to form ensemble network

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
X_test, y_test = load2d('TEST')
print(X_test.shape, y_test.shape)


def make_ensemble(n_members):
  models = list()
  for i in range(n_members):
    # load model
    filename = '/content/drive/MyDrive/data/model_' + str(i + 1) + '.h5'
    model = load_model(filename)
    # store in memory
    models.append(model)
  return models
def ensemble_predictions(X_test, members):
  yhats = [model.predict(X_test) for model in members]
  yhats = np.array(yhats)
  # sum across ensembles
  summed = np.sum(yhats, axis=0)
  labels_pred = np.argmax(summed,axis=1)
  return labels_pred
def evaluate_predictions(labels):
  labels_pred = ensemble_predictions(X_test, models)
  true_labels = np.argmax(y_test, axis=1)
  test_acc = accuracy_score(true_labels, labels_pred)
  matrix = multilabel_confusion_matrix(true_labels,labels_pred, labels=[6,5,4,3,2,1,0])
  print('Confusion matrix : \n',matrix)
  # classification report for precision, recall f1-score and accuracy
  print('Classification report : \n', classification_report(true_labels,labels_pred, labels=[6,5,4,3,2,1,0]))
  return test_acc

models = make_ensemble(num_of_models) 
test_accuracy= evaluate_predictions(y_test) 
print(test_accuracy)