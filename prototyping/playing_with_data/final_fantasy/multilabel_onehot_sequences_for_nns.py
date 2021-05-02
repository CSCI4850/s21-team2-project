"""Convert data to multilabel one hot encoded form.

An input vector is (1, timesteps) where timesteps represents the number of notes in
sequence after the ith note.
An output vector is (1, timesteps) in the same fashion.

Since a given column of an input/output vector is a <class 'str'> note
or chord, it must be mapped to an integer first then used to make
a one hot array of length corresponding to the number of values in the
map. Chords must be decomposed into their constituent strings in order
for the multilabel one-hot encoding to work.
"""

import pickle
import numpy as np
import os
from parse_chord_to_list_of_strings import parse_chord_to_list_of_strings
from constants import PICKLE_PATH, INPUT_TIMESTEPS, OUTPUT_TIMESTEPS


def multilabel_onehot_encode_datasets(X, y, str_to_int_chord):
    """One-hot labels multilabel datasets.

    In this case, it uses the mapping arg `str_to_int_chord` to map
    a string chord/note such as 'C4' to a note. Then this function uses
    that integer version of the chord as the index to hot in the one-hot
    multilabel matrix. This function will use multiple indices to hot
    if the string happens to be a chord.

    :param X: Array like of input matrix.
    :param y: Array like of output matrix.
    :param str_to_int_chord: <class 'dict'> that maps each string note
     to an integer.
    :return: <class 'tuple'> of <class 'numpy.ndarray'> containing
        one-hot labeled input and output arrays.
    """
    # Validate input
    if (np.array(X).shape[0] != np.array(y).shape[0]):
        raise ValueError(
            '`X` and `y` must have the same number of samples... i.e., 1 input vector has an associated output vector.')

    # Iterate by sample
    one_hot_input_matrix = []
    one_hot_output_matrix = []
    for song_ix in range(np.array(X).shape[0]):

        # LOG
        print(f'Labeling sequence number {song_ix}.')

        # IO vectors for a given sample
        input_vector = X[song_ix]
        output_vector = y[song_ix]

        # One hot vectors to be added to the one hot matrices
        one_hot_input_vector = multilabel_onehot_encode_a_vector(
            input_vector, str_to_int_chord)
        one_hot_output_vector = multilabel_onehot_encode_a_vector(
            output_vector, str_to_int_chord)

        # Append the onehot labeled vectors to the matrices
        one_hot_input_matrix.append(one_hot_input_vector)
        one_hot_output_matrix.append(one_hot_output_vector)

    # Return the matrices as <class 'numpy.ndarray'> objs
    return (np.array(one_hot_input_matrix), np.array(one_hot_output_matrix))


def multilabel_onehot_encode_a_vector(vector, str_to_int_chord):
    """Takes a vector and multilabel encodes it. 

    :param vector: 1D array-like
    :param str_to_int_chord: <class 'dict'> mapping string chord to 
        integer.
    :return: <class 'list'> of one-hot multilabel encoded elements
        (notes and chords) for a given song sequence
    """
    # The vector whose elements are one-hot multilabel arrays
    one_hot_vector = []

    # Number of unique categories for labeling
    num_categories = len(str_to_int_chord)

    # Encode elements
    for element in vector:

        # An encoded chord in the vector
        one_hot_element = None

        # Check if element is chord
        if(len(element) > 3):

            # Get the list of string chords
            extracted_chord = parse_chord_to_list_of_strings(element)

            # Get the indices of each constituent note of the chord
            # and add to a list to be used for encoding
            note_ix_list = []
            for note in extracted_chord:
                int_chord = str_to_int_chord[note]
                note_ix_list.append(int_chord)

            # Generate one-hot array
            one_hot_element = [0 for i in range(num_categories)]
            for i in note_ix_list:
                one_hot_element[i] = 1

        # Otherwise element is a note
        else:

            # Get the mapped integer, generate the array, and encode
            int_chord = str_to_int_chord[element]
            one_hot_element = [0 for i in range(num_categories)]
            one_hot_element[int_chord] = 1

        # Append the labeled element to the vector
        one_hot_vector.append(one_hot_element)

    # Return the constructed one-hot label element vector
    return one_hot_vector


def main():
    # Load mapping and chords datastructures
    with open(os.path.join(PICKLE_PATH, 'piano_note_encoding_dict'), 'rb') as fobj:
        encoding_dict = pickle.load(fobj)
    with open(os.path.join(PICKLE_PATH, f'piano_time_series_{INPUT_TIMESTEPS}_input_{OUTPUT_TIMESTEPS}_output_matrices_dict'), 'rb') as fobj:
        io_dict = pickle.load(fobj)

    # Get time series matrices and encoding dict
    input_matrix = io_dict['input_matrix']
    output_matrix = io_dict['output_matrix']
    str_to_int_chord = encoding_dict['str_to_int_chord']

    # Get the labeled matrices
    labeled_input_matrix, labeled_output_matrix = multilabel_onehot_encode_datasets(
        input_matrix,
        output_matrix,
        str_to_int_chord
    )

    # Pickle the labeled data
    with open(os.path.join(PICKLE_PATH, f'piano_time_series_LABELED_{INPUT_TIMESTEPS}_input_{OUTPUT_TIMESTEPS}_output_matrices_dict'), 'wb') as fobj:
        pickle.dump({'labeled_input_matrix': labeled_input_matrix,
                    'labeled_output_matrix': labeled_output_matrix}, fobj)

    # LOG
    print('Data labeled and pickle saved.')


if __name__ == '__main__':
    main()
