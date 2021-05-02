"""Encoding the data.

The chords datastructure is completely flattened (including) chords.
The unique notes will be found and mapped to an integer. The length
of this dictionary will be used to determine the number of classes
for multilabel one hot encoding. This is a more versatile scheme than
simply one-hot encoding an entire chord (which is a combination of distinct
notes). Therefore, input and output will be multilabel one-hot encoded.

E.g., the chord if the 4th octave in Cmajor is the only available octave,
and only natural notes are allwowed,
'C4D4E4' will be split to 'C4', 'D4', and 'E4' then one-hot encoded
as 
 A  B  C  D  E  F  G  
[0, 0, 1, 1, 1, 0, 0]
"""
import numpy as np
import pickle
import os
from parse_chord_to_list_of_strings import parse_chord_to_list_of_strings
from constants import PICKLE_PATH


def encode(chord_arr):
    """Returns encoding scheme (integer:string_note and vice versa)."""

    # Flatten the array
    flattened_arr = flatten(chord_arr)

    # Iterate through array, if len ele greater than 3 then it must be
    # a chord and that should be flattened, the chord should be deleted
    # And the flatend array elements should be appended to the end of the
    # flattened array
    ixs_of_chords_to_delete = []
    for ix, element in enumerate(flattened_arr):

        # If chord
        if (len(element) > 3):

            # Extract chord into list form
            parsed_chord = parse_chord_to_list_of_strings(element)

            # Append the extracted chord (just note strings) to the arr
            for note in parsed_chord:
                flattened_arr.append(note)

            # Chord needs to be removed
            ixs_of_chords_to_delete.append(ix)

    # Delete the chord indices
    for ix in sorted(ixs_of_chords_to_delete, reverse=True):
        del flattened_arr[ix]

    # Get the unique notes in the array
    unique_notes_arr = np.unique(flattened_arr)

    # Map index of unique note in flattened array to unique note
    # (e.g., {0: 'C4', 1: 'C5' ....})
    int_to_str_chord = {integer: string_note for integer,
                        string_note in enumerate(unique_notes_arr)}

    # Inverse mapping
    str_to_int_chord = {string_note: integer for integer,
                        string_note in int_to_str_chord.items()}

    return {'int_to_str_chord': int_to_str_chord, 'str_to_int_chord': str_to_int_chord}


def flatten(my_arr):
    """Flattens a list."""
    return [item for sublist in my_arr for item in sublist]


def main():
    # Get the features dictionary
    with open(os.path.join(PICKLE_PATH, 'piano_chord_duration_features_dict'), 'rb') as fobj:
        features_dict = pickle.load(fobj)

    # Get the encoding scheme dictionary
    encoding_scheme_dictionary = encode(features_dict['chords_ds'])

    # Write the encoding scheme
    with open(os.path.join(PICKLE_PATH, 'piano_note_encoding_dict'), 'wb') as fobj:
        pickle.dump(encoding_scheme_dictionary, fobj)

    # LOG
    print('Encoding complete.')


if __name__ == '__main__':
    main()
