"""Parse final fantasy music data.
env: vgm-data
"""

# Imports
from music21 import *
import mido
from IPython.display import display
import zipfile
import pickle
import numpy as np
import os
import sys
from constants import PICKLE_PATH


def make_chord_and_duration_datastructures(path, runtime_minimum_in_seconds=None, must_have_instrument=None):
    """Parse chords and durations of midi files to strings.

    Only songs whose runtime is greater than the provided runtime in seconds
    and that have the specified instrument will be parsed

    :param path: <class 'str'> of directory in which midi files will be
        processed.
    :param runtime_minimum_in_seconds: <class 'int'> min length of a song.
    :param must_have_instrument: <class 'str'> is an instrument that
        a score must have in order for it to be parsed.
    :return: <class 'dict'> of datastructure which contain values of only
        string
    """
    # Validate runtime
    if (not runtime_minimum_in_seconds):
        runtime_minimum_in_seconds = 60
    elif(runtime_minimum_in_seconds <= 0):
        raise ValueError(
            '`runtime_minimum_in_seconds` arg must be greater than 0')

    # Validate instrument
    if (not must_have_instrument):
        must_have_instrument = 'Piano'
    else:
        # Get the list of available instruments for music21
        all_instr_subclasses_list = [cls.__subclasses__()
                                     for cls in instrument.Instrument.__subclasses__()]
        all_instrument_subclasses_flattened = [
            item for sublist in all_instr_subclasses_list for item in sublist]
        all_instrument_names = [
            instr.__name__ for instr in all_instrument_subclasses_flattened]

        # Check whether the instrument string provided is valid
        if (must_have_instrument not in all_instrument_names):
            raise ValueError(
                '`must_have_instrument` is not an instrument that music21 supports.')

    # Make a list of fnames in the directory containing the FF music
    midi_fname_list = os.listdir(path)

    # Validate files in directory by checking for .mid files
    has_at_least_one_midi = False
    ix = 0
    while(not has_at_least_one_midi):
        if (midi_fname_list[ix].endswith('.mid')):
            has_at_least_one_midi = True
        ix += 1
    if (not has_at_least_one_midi):
        raise ValueError(
            '`path` arg must have at least one file of `.mid` type.')

    # Create sublists for chords/notes and duration to be returned
    # by this function.
    chords_ds = []
    durations_ds = []

    # Iterate through files in data directory
    for ix, midi_file in enumerate(midi_fname_list):

        # Try for opening mido file
        try:

            # Open the midi
            mid = mido.MidiFile(os.path.join(path, midi_file))

            # Song must have length greater than or equal to the supplied min
            song_length_in_seconds = mid.length
            if (song_length_in_seconds >= runtime_minimum_in_seconds):

                # Parse the file with music21
                score = converter.parse(os.path.join(path, midi_file))

                # Get piano score if possible
                try:

                    # Extract piano from midi
                    piano_score = instrument.partitionByInstrument(score)[
                        must_have_instrument]

                    # LOG
                    print(f'{midi_file} has Piano')

                    # All notes/chords in the score
                    notes_to_parse = piano_score.recurse()

                    # Song data structures
                    this_song_notes_and_chords = []
                    this_song_durations = []

                    # Iterate through each element of the score
                    for music21_obj in notes_to_parse:

                        if (isinstance(music21_obj, note.Note)):

                            # Append to this song's lists
                            this_song_notes_and_chords.append(
                                music21_obj.pitch.nameWithOctave)
                            this_song_durations.append(
                                music21_obj.duration.quarterLength)

                        elif (isinstance(music21_obj, chord.Chord)):

                            # Append to this song's list
                            # .normalOrder attr would work here, too
                            this_song_notes_and_chords.append(''.join(str(n)
                                                                      for n in music21_obj.pitches))
                            this_song_durations.append(
                                music21_obj.duration.quarterLength)

                    # Append songs to parent datastructures
                    chords_ds.append(this_song_notes_and_chords)
                    durations_ds.append(this_song_durations)

                except KeyError:

                    # Song does not have the desired instrument part
                    print(f'{midi_file} does not have {must_have_instrument}.')

        # Mid is type 2
        except ValueError:
            print(
                f'{midi_file} is a type 2 mid file and therefore playback cannot be computed.')

        # Problem reading file
        except (mido.midifiles.meta.KeySignatureError, IOError) as MidoErrors:
            print(f'Mido failed to process {midi_file}')

            # Return the data structures as a dict
    return {'chords_ds': chords_ds, 'durations_ds': durations_ds}


def main():
    # if ('--parse' in sys.argv[1:]):

    # LOG
    print('Parsing file by converting MIDI note/durations to strings:')
    print()

    # Extract features
    path = r'C:\Dev\python\CS4850\TEAM2_LOCAL_DATA\Final_Fantasy'
    ff_features_dict = make_chord_and_duration_datastructures(path)

    # Remove any songs with less than 100 notes
    for ix, song in enumerate(ff_features_dict['chords_ds']):
        if (len(song) < 100):
            del ff_features_dict['chords_ds'][ix]
            del ff_features_dict['durations_ds'][ix]
            print(
                f'Songs at index {ix} were removed and had length of {len(song)}')

    # Write those features to file
    with open(os.path.join(PICKLE_PATH, 'piano_str_chord_duration_features_dict'), 'wb') as fobj:
        pickle.dump(ff_features_dict, fobj)

    # LOG
    print('All done.')


if __name__ == '__main__':
    main()
