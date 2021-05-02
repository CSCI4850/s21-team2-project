"""Helper function for encoding."""


def parse_chord_to_list_of_strings(concatenated_chord):
    """Convert a concatenated chord to list of notes for music21 chord.

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
