
# coding: utf-8

# In[15]:

from keras.layers import Input, Lambda, merge, Dense, Dropout, BatchNormalization, Activation, Add, Concatenate
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam, RMSprop
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy.random as rng
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import random
import time
from tqdm import tqdm
from scipy.spatial.distance import euclidean, braycurtis, canberra, chebyshev, cosine, minkowski


EPOCHS = 200
BATCH_S = 100
PATIENCE = 10
MODEL_PATH = 'models/'.format(EPOCHS, BATCH_S)
MODEL_NAME = 'mlp{}_b{}_bn_euc_dist.h5'.format(EPOCHS, BATCH_S)
HISTORY_PATH = 'history/'
HISTORY_NAME = 'mlp{}_b{}_bn_euc_dist.h5'.format(EPOCHS, BATCH_S)
TRAIN_PATH = 'dataset/train/shuffled+duplicates.h5'
TEST_PATH = 'dataset/test/test.h5'
LR = 0.01


def save_dataframe(path, df):
    path_l = path.split('/')
    file_name = path_l[-1]
    del path_l[-1]

    if not os.path.exists('/'.join(path_l)):
        os.makedirs('/'.join(path_l))
    df.to_pickle(path)
    print 'file saved - {}'.format(path)

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def l1_distance(vects):
    x, y = vects
    return K.sum( K.abs(x - y), axis = 1, keepdims = True )

def cosine_distance(vects):
    x, y = vects
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Dense(400, input_shape=(input_dim,), activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(400, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(400, activation='relu'))
    return seq

def create_bn_network(input_dim):
    inp = Input(shape=(input_dim, ))
    dense1 = Dense(128)(inp)
    bn1 = BatchNormalization()(dense1)
    relu1 = Activation('relu')(bn1)

    dense2 = Dense(128)(relu1)
    bn2 = BatchNormalization()(dense2)
    res2 = Add()([relu1, bn2])
    relu2 = Activation('relu')(res2)

    dense3 = Dense(128)(relu2)
    bn3 = BatchNormalization()(dense3)
    res3 = Add()([relu2, bn3])
    relu3 = Activation('relu')(res3)

    feats = Concatenate()([relu3, relu2, relu1])
    bn4 = BatchNormalization()(feats)

    model = Model(inputs=inp, outputs=bn4)

    return model

input_dim = 400

base_network = create_bn_network(input_dim)

input_a = Input(shape=(input_dim,))
input_b = Input(shape=(input_dim,))

processed_a = base_network(input_a)
processed_b = base_network(input_b)


eucl_distance = Lambda(euclidean_distance,
                       output_shape=eucl_dist_output_shape)([processed_a, processed_b])

prediction = Dense(1, activation='sigmoid')(eucl_distance)

model = Model(inputs=[input_a, input_b], outputs=prediction)

opt = SGD(lr=LR)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', f1, recall, precision])
model.summary()


print ('Loading data...')
train_df = pd.read_pickle(TRAIN_PATH).sample(frac=1)
val_df  = pd.read_pickle(TEST_PATH).sample(frac=0.01)
print ('Sucessfully load...')


def generate_batch(x_left, x_right, y, size):
    while True:
        for i in xrange(0, len(x_left), size):
            x_left_b   = x_left[i:i+size]
            x_right_b  = x_right[i:i+size]
            y_b        = y[i:i+size]

            x_left_b   = np.array([np.array(row) for row in x_left_b])
            x_right_b  = np.array([np.array(row) for row in x_right_b])

            yield ([x_left_b, x_right_b], y_b)

def get_number_of_batches(dataset, b_size):
    ds_len = len(dataset)
    num_of_batches = int(round((1.0*len(dataset))/b_size))

    if ds_len % b_size:
        num_of_batches+=1

    return num_of_batches


x_left  = train_df.vec1.values
x_right = train_df.vec2.values
y_train = train_df.is_duplicate.values

x_left_val  = val_df.vec1.values
x_right_val = val_df.vec2.values
y_val       = val_df.is_duplicate.values


batch_total = get_number_of_batches(x_left, BATCH_S)
val_steps = get_number_of_batches(x_left_val, BATCH_S)

callbacks = [ModelCheckpoint(MODEL_PATH+MODEL_NAME, monitor='val_loss', save_best_only=False),
             EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1, mode='auto')
            ]

t0 = time.time()

history = model.fit_generator(generate_batch(x_left, x_right, y_train, BATCH_S),
                                steps_per_epoch=batch_total,
                                nb_epoch=EPOCHS,
                                verbose=1,
                                validation_data=generate_batch(x_left_val, x_right_val, y_val, BATCH_S),
                                validation_steps=val_steps,
                                callbacks = callbacks
                               )

t1 = time.time()
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))


summary_stats = pd.DataFrame({'epoch': [ i + 1 for i in history.epoch ],
                              'train_loss': history.history['loss'],
                              'valid_loss': history.history['val_loss'],
                              'valid_f1': history.history['val_f1'],
                              'valid_precision': history.history['val_precision'],
                              'valid_recall': history.history['val_recall'],
                              'train_recall': history.history['recall'],
                              'train_precision': history.history['precision'],
                              'train_f1': history.history['f1']})
save_dataframe(HISTORY_PATH+HISTORY_NAME, summary_stats)
