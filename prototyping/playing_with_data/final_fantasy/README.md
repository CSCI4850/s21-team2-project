# Constants

Pay attention to the constants in constants.py.

The PICKLE_PATH is particularly important.

# On Imports

https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path

# On Data Compression for Remote Repo

Pickled data should be zipped up due to it's size. Whenever files in the zip file need to be read,

they should be read in the program following the implementation in the link below:

https://stackoverflow.com/questions/10908877/extracting-a-zipfile-to-memory/10909016

# Order of Data Processing

These steps occur only once. Final fantasy data is framed a multilabel classification problem.

1. parse.py --> Gets string versions of chords/notes and durations in the midi dataset
2. integer_string_mapping.py --> Creates a dictionary maps an integer to each unique string NOTE in the data produced by parse.py

Repeat these with different hyperparameters.

1. get_time_series_sequence_for_nns.py --> Frames the data as a supervised learning problem where a number of input timesteps are used to predict a number of output timesteps
2. multilabel_onehot_sequences_for_nns.py --> Converts time series framed data produced from get_time_series_sequence_for_nns.py to a multilabeled one hot array where the number of categories is determined by the number of integer keys produced from the integer_string_mapping.py file.
3. train_test_holdoutval_split.py --> Uses sklearn.model_selection.TimeSeriesSplit to extract training, validation, and holdout validation sets.
