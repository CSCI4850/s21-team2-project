"""Loads a model and predicts music with it.

Note. For lstm-autoencoder only outputs fixed length sequence of songs,
this is sequence to sequence prediction.

On interpreting probabilities for multiclass classification problems:
https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
"""

# Imports
import os
from pathlib import Path
import tensorflow.keras as keras
from music21 import *
import pickle
import numpy as np
import datetime
from constants import PICKLE_PATH, INPUT_TIMESTEPS, OUTPUT_TIMESTEPS
import sys


def make_predictions(model, X_test, positive_classification_threshold):
    """Builds 2 onehot matrices: (1) generated song, (2) original song.

    :param model: keras model
    :param X_test: <class 'numpy.ndarray'> from which a random chord
        sequence will be selected.
    :param positive_classification_threshold: <class 'float'> that
        determines the minimum probability that an output
        neuron must have in order to be considered a positive classification
        for a particular category.
    :return: <class 'tuple'> of <class 'numpy.ndarray'>
    """
    # LOG
    print('Generating music')

    # Occurrence of failed threshold
    cnt_failed_threshold = 0

    # Take random starting starting point for validation
    random_ix_of_sequence_elem_in_x_test = np.random.randint(
        0, X_test.shape[0])

    # Variables to be modified in song generation loop
    generated_song = np.empty(shape=(1, 0, X_test.shape[2]))
    input_tensor = X_test[random_ix_of_sequence_elem_in_x_test].reshape(
        1, X_test.shape[1], X_test.shape[2])
    original_song = input_tensor.copy()

    # (?, output_timestep, categories)
    predicted_chords_tensor = model.predict(
        input_tensor, verbose=0)

    # A sample is (timesteps, categories) dimensional
    for sample in predicted_chords_tensor:

        # A sample has output timesteps determined by '..../playing_with_data/final_fantasy/get_time_series_sequence_for_nns'
        # A chord is (categories, ) dimensional
        for ix, chord_ in enumerate(sample):

            chord_ = onehot_label_nn_output(
                chord_, positive_classification_threshold)

            # Append the chord to the generated song but if not chord
            # is generated, just take a chord from the input sequence
            chord_ = chord_.reshape(1, 1, input_tensor.shape[2])
            if (np.amax(chord_) == 0):

                # If no classification meets the
                # positive_classification_threshold, then the one-hot
                # vector will be all 0s. Therefore, the output
                # will be estimated as the very last element in the
                # input sequence. Since the prediction_input_matrix
                # has dims (?, timestep, categories) then [-1][-1]
                # gets the last time step's chord represented by
                # a vector (categories,)
                filler_chord_from_input_tensor = input_tensor[0][ix].reshape(
                    1, 1, input_tensor.shape[2])
                generated_song = np.append(
                    generated_song, filler_chord_from_input_tensor, axis=1)
                cnt_failed_threshold += 1
            else:
                generated_song = np.append(
                    generated_song, chord_, axis=1)

    # Return the onehot multilabeled generated song of song_length
    # and the original song used to generate it
    print('Number of failed predictions:', cnt_failed_threshold)
    return (generated_song, original_song)


def onehot_label_nn_output(chord, positive_classification_threshold):
    """Takes a chord vector output from nn and multilabel one-hots it."""
    # Get the element of a chord (which is probability vector)
    # and hot a category based on probability threshold
    for ix, category in enumerate(chord):
        if (category > positive_classification_threshold):
            chord[ix] = 1
        else:
            chord[ix] = 0

    # Return the labeled chord
    return chord


def make_music21_stream(onehot_matrix, int_to_str_chord, instrument_part=None):
    """Converts one-hot matrices to writable songs.

    :param onehot_matrix: <class 'numpy.ndarray'> of shape
        (? =~ 1, timestep, classes) to be converted to string chords.
    :param int_to_str_chord: <class 'dict'> that maps integers to
        individual notes (still referred to as chords).
    :param instrument_part: <class 'music21.stream.instrument.Instrument'>
        to be used for the stream generation.
    """
    # Default instrument
    if (not instrument_part):
        instrument_part = instrument.KeyboardInstrument()

    # The music stream
    music21_stream = stream.Part()
    music21_stream.append(instrument_part)

    # Iterate through songs (should just be 1 song)

    for song in onehot_matrix:

        # A song will have some number of chords determined a priori
        for chord_ in song:

            # A string'ified musical element representing
            # a note or a chord
            musical_element = []
            for ix, category in enumerate(chord_):

                # Get the predicted musical element as a list of strings
                if (category == 1):
                    musical_element.append(int_to_str_chord[ix])

            # If the length of the musical element is 1 then the musical element
            # must be a NOTE otherwise it's a collection of NOTES aka a CHORD
            if (len(musical_element) == 1):
                music21_stream.append(note.Note(musical_element[0]))
            else:
                music21_stream.append(chord.Chord(musical_element))

    # Return the musical score
    return music21_stream


def parse_chord_to_list_of_strings(concatenated_chord):
    """Convert a concatenated chord to list of notes for music21 chord.

    Not in use for this file.

    :param concatenated_chord: <class 'str'>  A concatenated string
        representing a string.
    :return: <class 'list'> A list of strings representing a chord
        for music21
    """
    # Holds the final chord
    chord = []

    # Determines from where to slice a given chord
    slice_from = 0

    # Iterate through the strin (e.g., 'C4E4G4')
    for ix, char in enumerate(concatenated_chord):

        # Notes are delineated by a final digit representing the octatve
        # for that note. When a digit is encountered,
        # The note is the slice from the current slice to the current ix
        # plus one (because string slicing is exclusive)
        if char.isdigit():
            chord.append(concatenated_chord[slice_from:ix + 1])
            slice_from = ix + 1

    # Return list of notes (a chord)
    return chord


def main():
    # Testing hyperparams
    positive_classification_threshold = 0.2

    # Load a model
    cur_folder = os.path.dirname(os.path.realpath(__file__))
    model_path = None
    print(sys.argv)
    if (len(sys.argv) == 1):
        paths = sorted(Path(os.path.join(cur_folder, 'saved_models_h5')).iterdir(),
                       key=os.path.getmtime)
        model_path = paths[-1].__str__()
        print(f'Default file path (most recent saved) used: {model_path}')
    else:
        try:
            model_path = os.path.join(
                cur_folder, 'saved_models_h5', sys.argv[1])
            print(f'Specified file name: {model_path}')
        except:
            print('Error with filename')

    # Load the model
    model = keras.models.load_model(model_path)

    # Load the validation set from this dictionary
    with open(os.path.join(PICKLE_PATH, f'piano_train_val_holdval_split_for_time_series_LABELED_{INPUT_TIMESTEPS}_input_{OUTPUT_TIMESTEPS}_output_dict'), 'rb') as fobj:
        train_val_holdout_dict = pickle.load(fobj)

    # Load the mapping dictionary
    with open(os.path.join(PICKLE_PATH, 'piano_note_encoding_dict'), 'rb') as fobj:
        encoding_dict = pickle.load(fobj)

    # Assign the validation set
    X_validation = train_val_holdout_dict['X_validation']

    # Make predictions
    (onehot_generated_song, onehot_original_song) = make_predictions(
        model,
        X_validation,
        positive_classification_threshold=positive_classification_threshold,
    )

    # Make the music 21 objs
    int_to_str_chord = encoding_dict['int_to_str_chord']
    generated_song = make_music21_stream(
        onehot_generated_song,
        int_to_str_chord=int_to_str_chord,
    )
    original_song = make_music21_stream(
        onehot_original_song,
        int_to_str_chord=int_to_str_chord
    )

    # Write the music21 mids to file
    truncated_model_name = os.path.splitext(
        os.path.basename(os.path.normpath(model_path)))[0]
    now = datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')
    gen_path = os.path.join(cur_folder, 'generated_midis')
    generated_song.write(
        'midi', f'{gen_path}/{now}_threshold_is_{positive_classification_threshold}_for_{OUTPUT_TIMESTEPS}_note_generated_song_using_{truncated_model_name}.mid')
    original_song.write(
        'midi', f'{gen_path}/{now}_for_{INPUT_TIMESTEPS}_note_original_song.mid')

    # LOG
    print('Songs written. Test network complete.')


if __name__ == '__main__':
    main()
