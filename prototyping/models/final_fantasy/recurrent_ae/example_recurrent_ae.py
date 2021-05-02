import pickle
import os
import tensorflow.keras as keras
import datetime
from contextlib import redirect_stdout
import pickle

# CWD
cwd = os.path.dirname(os.path.realpath(__file__))

# Model
model = keras.models.Sequential()

model.add(keras.layers.LSTM(
    100, return_sequences=True, input_shape=[None, 28]))

model.add(keras.layers.LSTM(30))

model.add(keras.layers.RepeatVector(28))

model.add(keras.layers.LSTM(100, return_sequences=True))

model.add(keras.layers.TimeDistributed(
    keras.layers.Dense(28, activation='sigmoid')))


# Compile
model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=[keras.metrics.Precision(), keras.metrics.Recall()]
)

# Plot it
keras.utils.plot_model(model, to_file=os.path.join(
    cwd, './figures/example_lstm_ae.png'),
    expand_nested=True,
    show_shapes=True
)
