# spectra classifier
import numpy as np
from numpy.random import seed
seed(7)
import math
from numpy import array
from keras import initializers
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM, BatchNormalization, Activation
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import to_categorical
from keras.optimizers import Adam, SGD
from keras.losses import categorical_crossentropy
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def scheduler(epoch):
    lr = 1e-5
    if epoch < 5:
        return lr
    else:
        return lr * 0.1 

def scale_data(X):
    #scaler = MinMaxScaler()
    scaler = StandardScaler()
    #scaler.fit(X)
    #ret = scaler.transform(X)
    ret = np.empty(shape = X.shape)
    for idx,r in enumerate(X):
        norm = np.linalg.norm(r)
        scaled = r/norm
        ret[idx] = scaled
    return ret

def cce_function(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred, label_smoothing=0.1)

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
        for i in indices:
            batch.append(i)
            if len(batch)==batch_size:
                yield X[batch], Y[batch]
                batch=[]

def test_generator(X, Y, batch_size):
    indices = np.arange(len(X)) 
    batch=[]
    while True:
        for i in indices:
            batch.append(i)
            if len(batch)==batch_size:
                yield X[batch], Y[batch]
                batch=[]


train_fname = 'vector-data/train.csv'
raw_train = open(train_fname, 'rt')
train_data = np.loadtxt(raw_train, delimiter=",")
train_X = train_data[:,1:8193]
train_Y = train_data[:,8193]
train_X = scale_data(train_X)
train_Y = to_categorical(train_Y, num_classes=5)

val_fname = 'vector-data/val.csv'
raw_val = open(val_fname, 'rt')
val_data = np.loadtxt(raw_val, delimiter=",")
val_X = val_data[:,1:8193]
val_Y = val_data[:,8193]
val_X = scale_data(val_X)
val_Y = to_categorical(val_Y, num_classes=5)

test_fname = 'vector-data/test.csv'
raw_test = open(test_fname, 'rt')
test_data = np.loadtxt(raw_test, delimiter=",")
test_X = test_data[:,1:8193]
test_Y = test_data[:,8193]
test_X = scale_data(test_X)
test_Y = to_categorical(test_Y, num_classes=5)

# define model
k = 15
train_X = train_X.reshape((4511, 8192, 1))
val_X = val_X.reshape((1288, 8192, 1))
test_X = test_X.reshape((646, 8192, 1))
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=k, kernel_initializer='he_normal', padding='same', activation='relu', input_shape=(8192,1)))
model.add(Conv1D(filters=64, kernel_size=k, kernel_initializer='he_normal', padding='same', activation='relu'))
#model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=4, padding='same'))

model.add(Conv1D(filters=128, kernel_size=k, kernel_initializer='he_normal', padding='same', activation='relu'))
model.add(Conv1D(filters=128, kernel_size=k, kernel_initializer='he_normal', padding='same', activation='relu'))
#model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=4, strides=4))

model.add(Conv1D(filters=256, kernel_size=k, kernel_initializer='he_normal', padding='same', activation='relu'))
model.add(Conv1D(filters=256, kernel_size=k, kernel_initializer='he_normal', padding='same', activation='relu'))
model.add(Conv1D(filters=256, kernel_size=k, kernel_initializer='he_normal', padding='same', activation='relu'))
#model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=4, strides=4))

model.add(Conv1D(filters=512, kernel_size=k, kernel_initializer='he_normal', padding='same', activation='relu'))
model.add(Conv1D(filters=512, kernel_size=k, kernel_initializer='he_normal', padding='same', activation='relu'))
model.add(Conv1D(filters=512, kernel_size=k, kernel_initializer='he_normal', padding='same', activation='relu'))
#model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=4, strides=4))

model.add(Conv1D(filters=512, kernel_size=k, kernel_initializer='he_normal', padding='same', activation='relu'))
model.add(Conv1D(filters=512, kernel_size=k, kernel_initializer='he_normal', padding='same', activation='relu'))
model.add(Conv1D(filters=512, kernel_size=k, kernel_initializer='he_normal', padding='same', activation='relu'))
#model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=4, strides=4))


#model.add(LSTM(32))
#model.add(TimeDistributed(Dense(1)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(5, activation='softmax'))

#sgd = SGD(lr=1e-5, momentum=0.9)
opt = Adam(lr=1e-5, clipvalue=5)
model.compile(optimizer=opt, loss=cce_function, metrics=['accuracy'])
print(model.summary())

batch_size = 16
train_it = batch_generator(train_X, train_Y, batch_size)
val_it = val_generator(val_X, val_Y, batch_size)

lr_scheduler = LearningRateScheduler(scheduler)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=math.sqrt(0.1),
                              patience=5, min_lr=1e-8, verbose=1)
# fit model
model.fit_generator(train_it,
                    steps_per_epoch=4511//batch_size,
                    validation_data=val_it,
                    validation_steps=1288//batch_size,
                    epochs=50,
                    callbacks=[reduce_lr],
                    verbose=1)

val_it = val_generator(val_X, val_Y, 1)
Y_pred = model.predict_generator(val_it, 1288)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(val_data[:,8193], y_pred))


test_it = val_generator(test_X, test_Y, 1)
Y_pred = model.predict_generator(test_it, len(test_X))
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(test_data[:,8193], y_pred))



#model.fit(train_X, train_Y, epochs=300, verbose=0, batch_size=16)

# connect the encoder LSTM as the output layer
#model = Model(inputs=model.inputs, outputs=model.layers[0].output)

# get the feature vector for the input sequence
#yhat = model.predict(sequence)
#print(yhat.shape)
#print(yhat)
