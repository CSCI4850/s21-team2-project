"""Make predictions using a trained model."""

# Imports
# MIDI processing
from datetime import datetime
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from music21 import *

# Tensorflow and Keras
import tensorflow.keras as keras
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

# Sklearn
from sklearn.model_selection import train_test_split

# Wrangling
import numpy as np

# Visual
import matplotlib.pyplot as plt

# Misc
import pickle
from pathlib import Path
import os
from datetime import datetime


def unpickle(path_to_dict):
    """Unpickles a pickled file.

    :param path_to_dict: <class 'str'> of path to file
    :return: Unpickled python object
    """
    # LOG
    print('Unpickling data')

    # Unpickle
    with open(path_to_dict, 'rb') as fpath:
        return pickle.load(fpath)


def process_features(data_dict, sequence_length):
    """Returns train-test-validation sets.

    :param data_dict: <class 'dict'> containg keys related to
        encoding and features
    :param sequence_length: <class 'int'> of desired time-steps for
        model architecture
    """
    # LOG
    print('Processing features and extract train-test-holdout validation.')

    # Key:Values of data_dict
    chords_ds = data_dict['chords_ds']
    string_to_int_chord = data_dict['chord_to_int']

    # Flatten list
    chords_ds_flattened = [item for sublist in chords_ds for item in sublist]

    # Dimensions for LSTM
    n_sequence_patterns = None  # number of samples

    # Unique categories for a given sample
    n_vocab = len(np.unique(chords_ds_flattened))

    # Number of a features a given sample vector has (1 for this study)
    num_features = 1

    # Empty lists for train data
    network_input = []
    network_output = []

    # Sequence construction
    # The chord_ds encapsulates the chords/notes associated with a particular score index
    # create input sequences and the corresponding outputs
    for i in range(0, len(chords_ds_flattened) - sequence_length):

        # i to the i + sequence length (exclusive) input
        # and map the sequence input to int and append to network input
        chord_sequence_input = chords_ds_flattened[i:i + sequence_length]

        network_input.append([string_to_int_chord[chord]
                             for chord in chord_sequence_input])

        # the i + sequence length output (next note after a sequence of notes)
        # and map to int and append to network output
        chord_sequence_output = chords_ds_flattened[i + sequence_length]
        network_output.append(string_to_int_chord[chord_sequence_output])

    # Update number of sequence patterns based on network_input
    n_sequence_patterns = len(network_input)

    # Shape the input for LSTM (which take (?, t_timesteps, f_features))
    # where ? is the number of samples (pattern sequences)
    network_input = np.reshape(
        network_input, (n_sequence_patterns, sequence_length, 1))

    # Normalize input
    network_input = network_input / float(n_vocab)

    # One-hot label the network_output
    network_output = keras.utils.to_categorical(
        network_output, num_classes=n_vocab)

    # Define the train-test and holdout validation sets
    X_train_test, X_holdout_validation, y_train_test, y_holdout_validation = train_test_split(
        network_input,
        network_output,
        random_state=0,
        shuffle=False,
        test_size=0.2,
    )

    # Define the train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_test,
        y_train_test,
        random_state=0,
        shuffle=False,
        test_size=0.2,
    )

    # LOG
    # print()
    # print('Network input type:', type(network_input))
    # print('Network out type:', type(network_output))
    # print('X_train_test and y_train_test types:',
    #       type(X_train_test), type(y_train_test))

    return (X_train_test, X_holdout_validation,
            y_train_test, y_holdout_validation,
            X_train, X_test, y_train, y_test)


def generate_songs(model, sequence_length, X_test,
                   n_vocab, int_to_string_chord,
                   song_length):
    """Makes a generated song and the original song.

    :param model: keras model to be used for prediction.
    :param sequence_length: <class 'int'> of the second dimension of
        the model input (i.e., (?, timesteps=sequence_length, n_features))
    :param X_test: <class 'numpy.ndarray'> of chord sequences
    :param n_vocab: <class 'int'> of categories of a given feature.
    :param int_to_string_chord: <class 'dict'> of integers : chords (strings).
    :param song_length: <clas 'int'> of how many notes are desired in
        the predicted song.
    :return: <class 'tuple'> of <class 'music21.stream.Part'> where
        the first element is the generated song and the second element
        is the song the generated song was based on
    """
    # LOG
    print('Generating music')

    # Take random starting sequencet
    random_ix_of_sequence_elem_in_x_test = np.random.randint(
        0, X_test.shape[0])

    # Chord sequence is array of floats
    chord_sequence = X_test[random_ix_of_sequence_elem_in_x_test]

    # Original sequence is this initial array of floats
    original_sequence = chord_sequence.copy()

    # Integer predictions of chords
    predicted_sequence = []

    # Make song of variable length
    for note_index in range(song_length):

        # Reshape the input for the network
        # (?, sequence_length=..., 1 feature (a chord))
        prediction_input = np.reshape(chord_sequence, (1, sequence_length, 1))

        # Generate (sequence_length, 1) dimensional predictions
        # Which are probabilities of the next chord belonging to a
        # a particular chord
        prediction = model.predict(prediction_input, verbose=0)

        # The index of the argmax of the prediction is the chord (feature) with
        # highest probability of being classified (making logical music) due to
        # softmax activation
        index = np.argmax(prediction)

        # Append that chord to a list of predicted chords
        predicted_sequence.append(index)

        # Convert the index into a normalized value and append it to
        # the existing chord_sequence of floats
        chord_sequence = np.append(chord_sequence, (index / float(n_vocab)))

        # After the first prediction, the chord_sequence now hold 33 notes
        # Keep only the next set of notes (i.e., notes 1-33) instead of notes
        # (0-32)... basically just sliding window prediction...
        chord_sequence = chord_sequence[1:len(chord_sequence)]

    # Generate a music21 stream based on the predictions
    generated_song = generate_music21_stream_from_chords_arr(
        predicted_sequence,
        int_to_string_chord=int_to_string_chord
    )

    # Original song -- note the original sequence is all floats
    scaled_original_sequence = (
        (original_sequence * n_vocab).flatten()
    ).astype(int)
    original_song = generate_music21_stream_from_chords_arr(
        scaled_original_sequence,
        int_to_string_chord=int_to_string_chord
    )

    # Return
    return (generated_song, original_song)


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


def generate_music21_stream_from_chords_arr(chord_arr, int_to_string_chord=None, instrument_part=None):
    """Convert a list of chords to a music21 stream.

    Helper function for generating songs

    :param chord_list: Array like list of chords. If elements of array like
        are NOT strings, must provide the `chord_to_int` argument
    :param int_to_string_chord: <class 'dict'> that maps integer chords to
        string chords.
    :param instrument: <class 'music21.instrument.Instrument'> Defaults
        to KeyboardInstrument()
    :return: <class 'music21.stream.Part'>
    """

    # Check if incorrect args provided
    numpy_arr = np.array(chord_arr)
    numpy_arr_type = numpy_arr.dtype
    if (not (np.issubdtype(numpy_arr_type, np.integer) or issubclass(numpy_arr_type, np.character))):
        raise TypeError(
            "`chord_arr` must have <class `str`> or <class 'int'> like elements.")

    # Needs mapping arg...
    if ((int_to_string_chord is None) and np.issubdtype(numpy_arr_type, np.integer)):
        raise ValueError(
            "`chord_arr` is an array-like of <class 'int'> elements, so `in_to_string_chord` arg must be provided.")

    # Auto convert integer to string
    if (np.issubdtype(numpy_arr_type, np.integer) and (int_to_string_chord)):
        chord_arr = [int_to_string_chord[chord] for chord in chord_arr]

    # Set default instrument
    if (not instrument_part):
        instrument_part = instrument.KeyboardInstrument()

    # Make stream
    # Generate stream with Part as instrument
    generated_stream = stream.Part()
    generated_stream.append(instrument_part)

    # Append notes to stream
    for single_chord in chord_arr:
        try:
            chord_or_note_to_append = note.Note(single_chord)
            generated_stream.append(chord_or_note_to_append)
        except:
            chord_or_note_to_append = chord.Chord(
                parse_string_to_chord(single_chord)
            )
            generated_stream.append(chord_or_note_to_append)

    # Return the music21 stream
    return generated_stream


def main():

    # Sort list of directories in order of oldest to newest modification
    dir_folder = r'C:\Dev\python\CS4850\TEAM2_ORGANIZATION_REPO\models\lstm\saved_models_h5'
    paths = sorted(Path(dir_folder).iterdir(), key=os.path.getmtime)

    # Get the new modified saved model
    newest_saved_model_path = paths[-1].__str__()
    print('Model to load:', newest_saved_model_path)

    # Load that model
    model = keras.models.load_model(newest_saved_model_path)

    # Load the feature and encoding dictionary
    path_to_pickled_data = r'C:\Dev\python\CS4850\TEAM2_ORGANIZATION_REPO\pickled_data\pickled_tentatively_transposed_feature_and_encoding_dict'
    data_dict = unpickle(path_to_pickled_data)

    # Process data and get train test dictionary
    sequence_length = 100
    X_train_test, X_holdout_validation, y_train_test, y_holdout_validation, X_train, X_test, y_train, y_test = process_features(
        data_dict,
        sequence_length
    )

    # Make predictions
    n_vocab = y_test.shape[1]

    generated_song, original_song = generate_songs(
        model,
        sequence_length,
        X_test,
        n_vocab,
        int_to_string_chord=data_dict['int_to_chord'],
        song_length=32
    )

    # Write the songs to file
    now = datetime.now().strftime('%Y%m%d_%H-%S-%M')
    gen_path = r'C:\Dev\python\CS4850\TEAM2_ORGANIZATION_REPO\models\lstm\generated_songs' + \
        f'\\{now}'
    generated_song.write('midi', f'{gen_path}_generated_song.mid')
    original_song.write('midi', f'{gen_path}_original_song.mid')

    # LOG
    print('Songs written to file.')


if __name__ == '__main__':
    main()
