import numpy as np
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
import tensorflow as tf
from keras.optimizers import SGD, Adam

def cnn_1d(samples=None, labels=None, save_name=None, optimizer=Adam(lr=0.001), epochs=30, batch_size=64):
    model_inputs = keras.layers.Input(shape=(3001, 64))

    layer = keras.layers.Conv1D(filters=100, kernel_size=7, padding='same', activation='relu')(model_inputs)

    layer = keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same')(layer)

    layer = keras.layers.Conv1D(filters=150, kernel_size=5, padding='same', activation='relu')(layer)

    layer = keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same')(layer)

    layer = keras.layers.Conv1D(filters=200, kernel_size=3, padding='same', activation='relu')(layer)

    layer = keras.layers.GlobalMaxPooling1D()(layer)

    model_outputs = keras.layers.Dense(units=41, activation='softmax')(layer)

    model = keras.models.Model(inputs=model_inputs, outputs=model_outputs)

    model.summary()

    model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    keras.backend.get_session().run(tf.global_variables_initializer())

    model.fit(samples, labels, epochs=epochs, batch_size=batch_size)

    model.save(save_name)

def cnn_1d_dropout(samples=None, labels=None, save_name=None, optimizer=Adam(lr=0.001), epochs=30, batch_size=64):
    model_inputs = keras.layers.Input(shape=(3001, 64))

    layer = keras.layers.Conv1D(filters=100, kernel_size=7, padding='same', activation='relu')(model_inputs)

    layer = keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same')(layer)

    layer = keras.layers.Conv1D(filters=150, kernel_size=5, padding='same', activation='relu')(layer)

    layer = keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same')(layer)

    layer = keras.layers.Conv1D(filters=200, kernel_size=3, padding='same', activation='relu')(layer)

    layer = keras.layers.GlobalMaxPooling1D()(layer)

    layer = keras.layers.Dropout(0.2)(layer)

    model_outputs = keras.layers.Dense(units=41, activation='softmax')(layer)

    model = keras.models.Model(inputs=model_inputs, outputs=model_outputs)

    model.summary()

    model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    keras.backend.get_session().run(tf.global_variables_initializer())

    model.fit(samples, labels, epochs=epochs, batch_size=batch_size)

    model.save(save_name)

def cnn_1d_dense_dropout(samples=None, labels=None, save_name=None, optimizer=Adam(lr=0.001), epochs=30, batch_size=64):
    model_inputs = keras.layers.Input(shape=(3001, 64))

    layer = keras.layers.Conv1D(filters=100, kernel_size=7, padding='same', activation='relu')(model_inputs)

    layer = keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same')(layer)

    layer = keras.layers.Conv1D(filters=150, kernel_size=5, padding='same', activation='relu')(layer)

    layer = keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same')(layer)

    layer = keras.layers.Conv1D(filters=200, kernel_size=3, padding='same', activation='relu')(layer)

    layer = keras.layers.GlobalMaxPooling1D()(layer)

    layer = keras.layers.Dropout(0.2)(layer)

    layer = keras.layers.Dense(units=100, activation='relu')(layer)

    layer = keras.layers.Dropout(0.2)(layer)

    model_outputs = keras.layers.Dense(units=41, activation='softmax')(layer)

    model = keras.models.Model(inputs=model_inputs, outputs=model_outputs)

    model.summary()

    model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    keras.backend.get_session().run(tf.global_variables_initializer())

    model.fit(samples, labels, epochs=epochs, batch_size=batch_size)

    model.save(save_name)


if __name__ == '__main__':
    samples = np.load('1d_features.npy')
    labels = np.load('1d_labels.npy')
    samples = np.squeeze(samples)
    labels = np.squeeze(labels)

    if len(sys.argv) != 3:
        print('Please give correct inputs')
        sys.exit(1)
    model_name = sys.argv[1]
    save_name = sys.argv[2]
    if model_name == 'cnn_1d':
        cnn_1d(samples=samples, labels=labels, save_name=save_name)
    elif model_name == 'cnn_1d_dropout':
        cnn_1d_dropout(samples=samples, labels=labels, save_name=save_name)
    elif model_name == 'cnn_1d_dense_dropout':
        cnn_1d_dense_dropout(samples=samples, labels=labels, save_name=save_name)
    else:
        print('The model does not exist...')
        sys.exit(1)

