"""Split data."""

# Imports
import pickle
import os
from sklearn.model_selection import train_test_split
from constants import PICKLE_PATH, INPUT_TIMESTEPS, OUTPUT_TIMESTEPS


def main():
    # Load the labeled input output matrix data
    data_path = os.path.join(
        PICKLE_PATH, f'piano_time_series_LABELED_{INPUT_TIMESTEPS}_input_{OUTPUT_TIMESTEPS}_output_matrices_dict')
    with open(data_path, 'rb') as fobj:
        label_io_dict = pickle.load(fobj)

    # Extract matrices and the encoding
    input_matrix = label_io_dict['labeled_input_matrix']
    output_matrix = label_io_dict['labeled_output_matrix']

    # Extract training, testing, and holdout validation sets
    X_train_validation, X_holdout_validation, y_train_validation, y_holdout_validation = train_test_split(
        input_matrix, output_matrix, test_size=0.2, random_state=0, shuffle=False)

    # Extract just train and validation data
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation, y_train_validation, test_size=0.2, random_state=0, shuffle=False)

    # Inspect shape of each
    print('X_train, validation, and holdout validation shapes:',
          X_train.shape, X_validation.shape, X_holdout_validation.shape)
    print('y_train and holdout validation shaeps:',
          y_train.shape, y_validation.shape, y_holdout_validation.shape)

    # Save the split data
    with open(os.path.join(PICKLE_PATH, f'piano_train_val_holdval_split_for_time_series_LABELED_{INPUT_TIMESTEPS}_input_{OUTPUT_TIMESTEPS}_output_dict'), 'wb') as fobj:
        pickle.dump({
            'X_train': X_train,
            'X_validation': X_validation,
            'X_holdout_validation': X_holdout_validation,
            'y_train': y_train,
            'y_validation': y_validation,
            'y_holdout_validation': y_holdout_validation,
        }, fobj)

    # Save only the holdout validation data
    with open(os.path.join(PICKLE_PATH, f'piano_holdval_array_for_time_series_LABELED_{INPUT_TIMESTEPS}_input_{OUTPUT_TIMESTEPS}_output'), 'wb') as fobj:
        pickle.dump(X_holdout_validation, fobj)

    # LOG
    print('Split datasets pickled. Done.')


if __name__ == '__main__':
    main()
