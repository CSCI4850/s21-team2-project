"""The first processing step for time series data.

The chords_ds holds the notes/chords for each of the FF songs
in the available data set. To train a network, I will first simplify
the problem by flattening out the chords data structure entirely 
to remove boundaries between songs. Since this is a LSTM, and I can
take advantage of what is essentially time series data, I must shape
the data into the form of (n_samples, t_timesteps, f_features).
The data should be split into train-test and holdout validation sets.
The data for each set should be converted to multilabel one-hot vectors
where each category of a vector represents a particular note and a vector
with multiple hot categories represents a particular chord. The number
of categories is determined by the length of the values of the dictionary
that maps a given note (<class 'str'>) to a particular integer. The 
keys of the dictionary range from 0 to n where is n is the number of unique
notes (categories) across ALL songs in the data set. Therefore, a multilabel
one-hot vector will have n columns.

Consideration: Keep input/output vectors partitioned by song instead
of aggregated together.


Visualizing time step framing:
Each element is a value of a feature at some timestep t to the nth 
timestep. Let's say I want to use 4 timesteps to predict the next 1 timestep...

Data ->
i=0 1  2  3  4  5  6  7
[1, 2, 3, 4, 5, 6, 7, 8]

Single input vector x which is a part of the input matrix X
i      i+ts
[1,2,3,4]
Output vector y which is part of the output matrix Y
[5]

then
i+1     i+1+ts
[2,3,4,5]
[6]

then
i+2     i+2+ts
[3,4,5,6]
[7]

then
i+3     i+2+ts
[4,5,6,7]
[8]

therefore the range of i is limited by the size of the timestep. What if
I want multiple output? Let's say I want to use the 4 timesteps to 
output the next 2 timesteps

0=i      i+ts
[1,2,3,4]
Output vector y which is part of the output matrix Y

i+ts+1
[5, 6]

then
1=i+1     i+1+ts
[2,3,4,5]
[6, 7]

then
2=i+2     i+2+ts
[3,4,5,6]
[7, 8]

The range of i decreases by the number of desired output. Therefore,
since the desired timesteps is 4, and the desire number of outputs is 
2, then the range of i decreases by 2

therefore,
input = [ ]
output = [] 
for i in range(0, 8 - 4 - 2):
    x = [i     : i + 4]
    y = [i + 4 : i + 4 + 2]
    input.append(x)
    output.append(y)
    

For multilabel classification
https://towardsdatascience.com/multi-label-classification-and-class-activation-map-on-fashion-mnist-1454f09f5925
"""

import os
import sys
import pickle
from constants import PICKLE_PATH, INPUT_TIMESTEPS, OUTPUT_TIMESTEPS


def make_timeseries_input_and_output_matrices(data, input_timesteps, output_timesteps=None):
    """Takes data and preps it for timeseries problems

    :param data: Nested array-like [[song1], [song2]...]
    :param input_timesteps: <class 'int'> The number of timesteps that
        will be used for an input vector.
    :param output_timesteps: <class 'int'> The number of timesteps that
        will be used for an output (target) vector.
    """

    # Set default
    if (not output_timesteps):
        output_timesteps = 1

    # Validate input and output timesteps
    if ((input_timesteps <= 0) or (output_timesteps <= 0)):
        raise ValueError(
            '`input_timesteps` or `output_timesteps` must be greater than 0')

    # To be returned
    input_matrix = []
    output_matrix = []

    # Iterate through each song
    for ix, song in enumerate(data):

        # LOG
        print(f'About to iterate through song {ix}.')

        # Iterate through data and slice data
        if (len(song) <= input_timesteps or len(song) <= output_timesteps):
            print(
                'Cannot iterate. Song is shorter than the desired `input_timesteps` or `output_timesteps`')
        else:
            for timestep in range(0, len(song) - input_timesteps - output_timesteps):

                # Slice input and output vectors
                input_vector = song[timestep: timestep + input_timesteps]
                output_vector = song[timestep + input_timesteps: timestep +
                                     input_timesteps + output_timesteps]

                # Append to input and output matrices
                input_matrix.append(input_vector)
                output_matrix.append(output_vector)

        # LOG
        print(f'Iteration through song {ix} completed.')

    # Return the new matrices
    return (input_matrix, output_matrix)


def main():
    # Load chord data structures
    with open(os.path.join(PICKLE_PATH, 'piano_str_chord_duration_features_dict'),
              'rb') as fobj:
        data_dict = pickle.load(fobj)

    # Call function
    input_matrix, output_matrix = make_timeseries_input_and_output_matrices(
        data_dict['chords_ds'],
        input_timesteps=INPUT_TIMESTEPS,
        output_timesteps=OUTPUT_TIMESTEPS
    )

    # Pickle the data
    with open(os.path.join(PICKLE_PATH, f'piano_time_series_{INPUT_TIMESTEPS}_input_{OUTPUT_TIMESTEPS}_output_matrices_dict'), 'wb') as fobj:
        pickle.dump({'input_matrix': input_matrix,
                    'output_matrix': output_matrix}, fobj)

    # LOG
    print('Done processing data')


if __name__ == '__main__':
    main()
