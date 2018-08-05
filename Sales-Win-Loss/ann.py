# Lab 2 - Load the data
import os
import pandas as pd
import numpy as np


basepath='.'
outpath=os.path.join(basepath,"out")
if not os.path.exists(outpath):
    os.makedirs(outpath)

#df = pandas.read_csv( 'WA_Fn-UseC_-Sales-Win-Loss.csv')
    
df = pd.read_csv(os.path.join(basepath,'WA_Fn-UseC_-Sales-Win-Loss.csv'),sep = ",")
seed = 7
np.random.seed(seed)

# Lab 3 - Data Processing and selecting the correlated data
from sklearn import preprocessing 
le = preprocessing.LabelEncoder()
df['Opportunity Result'] = le.fit_transform(df['Opportunity Result'])
df['Supplies Group'] = le.fit_transform(df['Supplies Group'])
df['Supplies Subgroup'] = le.fit_transform(df['Supplies Subgroup'])
df['Region'] = le.fit_transform(df['Region'])
df['Competitor Type'] = le.fit_transform(df['Competitor Type'])


correlations = df.corr()['Opportunity Result']
correlations = correlations.sort_values()

# Lab 4 - Encoding and scaling the data
#df['Route To Market'] = le.fit_transform(df['Route To Market'])
df.drop(columns=['Opportunity Amount USD','Supplies Subgroup','Region','Supplies Group','Client Size By Employee Count','Client Size By Revenue','Elapsed Days In Sales Stage','Competitor Type','Opportunity Number'])
correlations = df.corr()['Opportunity Result']
df = pd.concat([pd.get_dummies(df['Route To Market'], prefix='Route To Market', drop_first=True),df], axis=1)
df = df.drop(columns=['Route To Market'])
correlations = correlations.sort_values()

y = df.iloc[:, df.columns.get_loc('Opportunity Result')].values

X = df.drop(columns=['Opportunity Result']).iloc[:, 0:df.shape[1] - 1].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Lab 5 - Create the ANN
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=8,activation='relu',input_dim=X.shape[1],name='Input_Layer'))
model.add(Dense(units=8,activation='relu',name='Hidden_Layer_1'))
model.add(Dense(units=1,activation='sigmoid',name='Output_Layer'))
model.compile(optimizer='nadam',loss = 'binary_crossentropy', metrics=['accuracy'])
# Lab 6 Tensorboard and fitting the model to the data
embedding_layer_names = set(layer.name for layer in model.layers)


import time
from time import strftime
from keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir=os.path.join (os.path.join (basepath, "logs"), format(strftime("%Y %d %m %H %M %S", time.localtime()))), write_graph=True, write_grads=True, write_images=True, embeddings_metadata=None, embeddings_layer_names=embedding_layer_names)

#Fit the ANN to the training set
model.fit(X, y, validation_split = .20, batch_size = 64, epochs = 25, callbacks = [tensorboard])

# summary to console
print (model.summary())
# Lab 7 Improve your model


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
   model = Sequential()
   model.add(Dense(units = 24, activation = 'relu', input_dim=X.shape[1], name= 'Input_Layer'))
   model.add(Dense(units = 24, activation = 'relu', name= 'Hidden_Layer_1'))
   model.add(Dense(1, activation = 'sigmoid', name= 'Output_Layer'))
   model.compile(optimizer= optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
   return model
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [64],
              'epochs': [25],
              'optimizer': ['adam','sgd','adamax','nadam']}

grid_search = GridSearchCV(estimator = classifier,
   param_grid = parameters,
   scoring = 'accuracy',
   verbose = 5)
grid_search = grid_search.fit(X, y)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


# Lab 8 Save your model


model_json = model.to_json()
with open(os.path.join (outpath, "model.json"), "w") as json_file:
    json_file.write(model_json)

import pickle
with open(os.path.join (outpath, "standardscalar.pickle"), 'wb') as handle:
    pickle.dump(sc, handle, protocol=pickle.HIGHEST_PROTOCOL)


model.save_weights(os.path.join (outpath, "model.h5"))

new_prediction = model.predict(sc.transform(np.array([[0,1,0,0,9,119,97,0,.81,.644,.245,6]])))
if (new_prediction > 0.5):
    print ("Won")
else:
    print ("Loss")