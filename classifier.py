# spectra classifier
import numpy as np
from numpy import array
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Flatten
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import to_categorical
from keras.optimizers import Adam, SGD
from keras.losses import categorical_crossentropy
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

def cce_function(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred, label_smoothing=0.1)

def scale_data(X):
    scaler = MinMaxScaler()
    scaler.fit(X)
    #ret = np.empty(shape = X.shape)
    #for idx,r in enumerate(X):
    #    norm = np.linalg.norm(r)
    #    scaled = r/norm
    #    ret[idx] = scaled
    return scaler.transform(X)

def batch_generator(X, Y, batch_size):
    indices = np.arange(len(X)) 
    batch=[]
    while True:
        # it might be a good idea to shuffle your data before each epoch
        np.random.shuffle(indices) 
        for i in indices:
            batch.append(i)
            if len(batch)==batch_size:
                #train_batch = np.empty(shape = X[batch].shape)
                #for idx,s in enumerate(X[batch]):
                #    noise = np.random.normal(s.min(),s.max(),21)
                #    train_batch[idx] = s+noise.reshape(21,1)
                yield X[batch], Y[batch]
                batch=[]

def val_generator(X, Y, batch_size):
    indices = np.arange(len(X)) 
    batch=[]
    while True:
        # it might be a good idea to shuffle your data before each epoch
        for i in indices:
            batch.append(i)
            if len(batch)==batch_size:
                yield X[batch], Y[batch]
                batch=[]

train_fname = 'data/train.csv'
raw_train = open(train_fname, 'rt')
train_data = np.loadtxt(raw_train, delimiter=",")
train_X = train_data[:,1:22]
train_Y = train_data[:,22]
#train_X = scale_data(train_X)
train_Y = to_categorical(train_Y, num_classes=5)

val_fname = 'data/val.csv'
raw_val = open(val_fname, 'rt')
val_data = np.loadtxt(raw_val, delimiter=",")
val_X = val_data[:,1:22]
val_Y = val_data[:,22]
#val_X = scale_data(val_X)
val_Y = to_categorical(val_Y, num_classes=5)

# define model
train_X = train_X.reshape((4603, 21, 1))
val_X = val_X.reshape((1315, 21, 1))
model = Sequential()
model.add(LSTM(192, input_shape=(21,1), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(192))
model.add(Dropout(0.5))
#model.add(TimeDistributed(Dense(1)))
#model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(5, activation='softmax'))
'''
model = Sequential()
model.add(Dense(128, input_dim=21, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='sigmoid'))
'''

#sgd = SGD(lr=1e-5, momentum=0.9)
opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=opt, loss=cce_function, metrics=['accuracy'])
print(model.summary())

batch_size = 16
train_it = batch_generator(train_X, train_Y, batch_size)
val_it = val_generator(val_X, val_Y, batch_size)
# fit model
model.fit_generator(train_it,
                    steps_per_epoch=4603//batch_size,
                    validation_data=val_it,
                    validation_steps=1315//batch_size,
                    epochs=100,
                    verbose=1)

Y_pred = model.predict_generator(val_it, 1315 // batch_size)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(val_data[:,22][:1312], y_pred))



#model.fit(train_X, train_Y, epochs=300, verbose=0, batch_size=16)

# connect the encoder LSTM as the output layer
#model = Model(inputs=model.inputs, outputs=model.layers[0].output)

# get the feature vector for the input sequence
#yhat = model.predict(sequence)
#print(yhat.shape)
#print(yhat)
