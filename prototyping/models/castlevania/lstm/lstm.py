# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from datetime import datetime
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from IPython import get_ipython

# %% [markdown]
# # Notes and Resources
# * Towards data science tutorial
#     * https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
# * Encompassing tutorial
#     * https://www.datacamp.com/community/tutorials/using-tensorflow-to-compose-music
# * LSTM for Time Series
#     * https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/
# * Model Checkpoint and Early Stopping
#     * https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
# * Tips for Improving RNNs
#     * https://danijar.com/tips-for-training-recurrent-neural-networks/
# * Alternative form of labeling for efficiency and scalability?
#     * https://datascience.stackexchange.com/questions/24729/one-hot-encoding-alternatives-for-large-categorical-values
#
# # Framing the problem
# Generation of music is a multiclass classification problem because a unique chord may be represented by a one-hot vector of chords where the length of the one-hot vector is the 'musical space' (vocab) of the particular classification problem.

# %%
# Imports

# MIDI processing
from music21 import *

# Tensorflow and Keras
import tensorflow.keras as keras
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
# get_ipython().run_line_magic('load_ext', 'tensorboard')

# Sklearn

# Wrangling

# Visual

# Misc


# %%
def parse_string_to_chord(concatenated_chord):
    """Convert a concatenated chord to list of notes for music21 chord.

    :param concatenated_chord: <class 'str'>  A concatenated string
        representing a string.
    :return: <class 'list'> A list of strings representing a chord
        for music21
    """
    chord = []
    slice_from = 0
    for ix, char in enumerate(concatenated_chord):
        if char.isdigit():
            chord.append(concatenated_chord[slice_from:ix + 1])
            slice_from = ix + 1

    # Return list of notes (a chord)
    return chord


def generate_music21_stream_from_int_chords(chord_list, mapping, instrument_part=None):
    """Convert a list of chords to a music21 stream.

    :param chord_list: Array like list of integer chords
        chords to be converted to music21 stream.
    :param mapping: <class 'dict'> that maps integer chords to 
        string chords.
    :param instrument: <class 'music21.instrument.Instrument'> Defaults
        to KeyboardInstrument()
    :return: <class 'music21.stream.Part'>
    """

    # Default instrument
    if (not instrument_part):
        instrument_part = instrument.KeyboardInstrument()

    # Map to string list
    chord_list = [mapping[chord] for chord in chord_list]

    # Make stream
    # Generate stream with piano as instrument
    generated_stream = stream.Part()
    generated_stream.append(instrument_part)

    # Append notes to stream
    for single_chord in chord_list:
        try:
            generated_stream.append(note.Note(single_chord))
        except:
            extracted_chord = parse_string_to_chord(single_chord)
            generated_stream.append(chord.Chord(extracted_chord))

    # Return the music21 stream
    return generated_stream

# %% [markdown]
# # Loading and preprocessing data


# %%
# Load feature and encoding data
path_to_pickled_data = '../../pickled_data/pickled_tentatively_transposed_feature_and_encoding_dict'
# with open(os.path.join(path_to_pickled_data, 'pickled_feature_and_encoding_dict'),'rb') as fobj:
#     data_dict = pickle.load(fobj)

with open(path_to_pickled_data, 'rb') as fobj:
    data_dict = pickle.load(fobj)

# Get Data
print(data_dict.keys())
chords_ds = data_dict['chords_ds']
durations_ds = data_dict['durations_ds']
chord_to_int = data_dict['chord_to_int']
duration_to_int = data_dict['duration_to_int']
int_to_chord = data_dict['int_to_chord']
int_to_duration = data_dict['int_to_duration']


# %%
# Flatten list
chords_ds_flattened = [item for sublist in chords_ds for item in sublist]
print(chords_ds_flattened[:10])
print("Flattend chords_ds type is <class 'str'>?", all(
    isinstance(x, str) for x in chords_ds_flattened))
print('Flattend chords_ds length:', len(chords_ds_flattened))
print('Unique notes/chords', len(chord_to_int))


# %%
# get_ipython().run_cell_magic('time', '', '\n# Data preprocessing\n\n## Defining training data -- previous sequential data\n\n# Dimensions for LSTM\nn_sequence_patterns = None  # number of samples\nn_vocab = len(np.unique(chords_ds_flattened))  # Unique categories for a given sample \nsequence_length = 100        # Number of time steps in a sample\nnum_features = 1    # Number of a features a given sample vector has (in this case only 1 feature which is an integer representing a chord/note)\n\n# Empty lists for train data\nnetwork_input = []\nnetwork_output = []\n\n## Sequence construction\n# The chord_ds encapsulates the chords/notes associated with a particular score index\n# create input sequences and the corresponding outputs\nfor i in range(0, len(chords_ds_flattened) - sequence_length):\n    chord_sequence_input = chords_ds_flattened[i:i + sequence_length]  # i to the i + sequence length (exclusive) input\n    chord_sequence_output = chords_ds_flattened[i + sequence_length]   # the i + sequence length output (next note after a sequence of notes)\n    network_input.append([chord_to_int[chord] for chord in chord_sequence_input])\n    network_output.append(chord_to_int[chord_sequence_output])\n\n# Update number of sequence patterns based on network_input\nn_sequence_patterns = len(network_input)')


# %%
# Shape the input for LSTM (which take (?, t_timesteps, f_features)) where ? is the number of samples (pattern sequences)
network_input = np.reshape(
    network_input, (n_sequence_patterns, sequence_length, 1))
# Divide each element of integer network array by n_vocab = 202 to normalize the input
network_input = network_input / float(n_vocab)

# Inspect the input
display('Input shape:', network_input.shape)
display(
    f'Sample input sequence (i.e. of shape (1 sample, t=5 out of {sequence_length} timesteps, 1 feature -- the chord)):', network_input[0][:5])

# One-hot label categorical data
network_output = keras.utils.to_categorical(
    network_output, num_classes=n_vocab)

# Inspect output
display('Output shape:', network_output.shape)
display(
    f'The expected note after the preceding {sequence_length} notes:', network_output[0])


# %%
# Divide sets
# X_holdout_validation and y_holdout_validation are not used until a model is optimized
X_train_test, X_holdout_validation, y_train_test, y_holdout_validation = train_test_split(
    network_input,
    network_output,
    random_state=0,
    shuffle=False,
    test_size=0.2,
)


# Inspect after sklearn function
print('Train-test/Holdout validation array:')
print(X_train_test.shape)
print(y_train_test.shape)
print(X_holdout_validation.shape)
print(y_holdout_validation.shape)

# Train test split (test will be used for validation=[] for hyperparam optimization)
X_train, X_test, y_train, y_test = train_test_split(
    X_train_test,
    y_train_test,
    random_state=0,
    shuffle=False,
    test_size=0.2,
)

# Inspect now
print()
print('Train/Test Arrays:')
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Check n_vocab
print()
print('n_vocab:', n_vocab)


# %%
# Build the model

# Hyperparameters
lstm_hidden_units = 256
dense_hidden_units = 128
batch_size = 512
epochs = 128

# Save current time
now = datetime.now().strftime('%Y%m%d_%H-%M-%S')

# Instantiate the sequential model
model = keras.models.Sequential()

# Input layer
model.add(
    keras.layers.LSTM(
        lstm_hidden_units,
        batch_input_shape=(
            batch_size, network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True,
        name='input_lstm'
    ))

# LSTM hidden 0
model.add(keras.layers.LSTM(lstm_hidden_units,
          return_sequences=True, recurrent_dropout=0.3))

# LSTM hidden 1
model.add(keras.layers.LSTM(lstm_hidden_units))

# Batch norm
model.add(keras.layers.BatchNormalization())

# Dropout
model.add(keras.layers.Dropout(0.3))

# Dense
model.add(keras.layers.Dense(dense_hidden_units,))
model.add(keras.layers.Activation('relu'))

# Batch norm
model.add(keras.layers.BatchNormalization())

# Dropout
model.add(keras.layers.Dropout(0.3,))

# Output
model.add(keras.layers.Dense(n_vocab, name='dense_output'))
model.add(keras.layers.Activation('softmax'))

# Compile it
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# Display it
keras.utils.plot_model(
    model, to_file=f'./figures/{now}_lstm.png', expand_nested=True, show_shapes=True)
model.summary()


# %%
get_ipython().run_cell_magic('time', '',
                             '\n### Fitting the model\n## Callbacks\nfit = True\nif (fit):\n    checkpoint = ModelCheckpoint(f\'./saved_models_h5/{now}_best_model.h5\', monitor=\'loss\', mode=\'min\', save_best_only=True, verbose=1)\n    early_stopping = EarlyStopping(monitor=\'loss\', verbose=1, patience=12)\n    #log_dir = "logs\\\\fit\\\\" + datetime.now().strftime("%Y%m%d-%H%M%S")\n    #tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)\n\n    ## Training\n    history = model.fit(\n        X_train,\n        y_train,\n        epochs=epochs,\n        batch_size=batch_size,\n        callbacks=[checkpoint, early_stopping],\n    )')


# %%
# Load a trained model if possible
if (not fit):
    model = keras.models.load_model('./saved_models_h5/best_model.h5')


# %%
# Take random starting from test set () not holdout validation set
random_ix_of_sequence_elem_in_x_test = np.random.randint(0, X_test.shape[0])
chord_sequence = X_test[random_ix_of_sequence_elem_in_x_test]
original_sequence = chord_sequence.copy()

# Inspect chord_sequence
print(original_sequence.shape)
print((original_sequence * n_vocab).astype(int))


# %%
get_ipython().run_cell_magic('time', '', '\n# Store predictions\nprediction_output = [] # generate desired number of notes/chords\nfor note_index in range(64):\n\n    # Reshape the input for the network (?, sequence_length=..., 1 feature (a chord))\n    prediction_input = np.reshape(chord_sequence, (1, sequence_length, 1))\n\n    # Generate (sequence_length, 1) dimensional song \n    prediction = model.predict(prediction_input, verbose=0)\n\n    # The index of the argmax of the prediction is the chord (feature) with\n    # highest probability of being classified (making logical music) due to\n    # softmax activation    \n    index = np.argmax(prediction)\n\n    # Map the result to a chord\n    result = int_to_chord[index]\n\n    # Append that chord to a list of predicted chords\n    prediction_output.append(result)    \n    \n    # Convert the result into a normalized value and append it to the existing chord_sequence\n    chord_sequence = np.append(chord_sequence, (index / float(n_vocab)))\n\n    # After the first prediction, the chord_sequence now hold 33 notes\n    # Keep only the next set of notes (i.e., notes 1-33) instead of notes (0-32)\n    # Sliding window prediction...\n    chord_sequence = chord_sequence[1:len(chord_sequence)]')


# %%
# Convert the chord_sequence back to integers
chord_sequence = (chord_sequence * n_vocab).astype(int)
chord_sequence


# %%
# Save the generated song
generated_song = generate_music21_stream_from_int_chords(
    chord_sequence, mapping=int_to_chord)

# Save the original song


# %%
# Save the generated song
print(now)
if (not fit):
    now = datetime.now().strftime('%Y%m%d_%H-%S-%M')
generated_song.write('midi', f'./generated_songs/{now}_generated_song.mid')


# %%
