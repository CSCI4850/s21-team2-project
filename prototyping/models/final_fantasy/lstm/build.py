"""Construct LSTM."""

# Imports
import pickle
import os
import tensorflow.keras as keras
import datetime
from contextlib import redirect_stdout
import pickle


def main():
    # Path to pickle data
    pickle_path = r'C:\Dev\python\CS4850\TEAM2_ORGANIZATION_REPO\playing_with_data\final_fantasy\pickled'

    # Load the labeled input output matrix data
    data_path = os.path.join(
        pickle_path, 'piano_train_val_holdval_split_dict')
    with open(data_path, 'rb') as fobj:
        train_test_holdoutval_dict = pickle.load(fobj)

    ###################
    # Build the model #
    ###################

    # Build Hyperparameters
    lstm_hidden_units = 256
    dense_hidden_units = 256

    # Shape of data
    X_shape = train_test_holdoutval_dict['X_train_validation'].shape
    y_shape = train_test_holdoutval_dict['y_train_validation'].shape

    # Save current time
    now = datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')

    # Instantiate the sequential model
    name = f'{now}_ff_lstm'
    model = keras.models.Sequential(name=name)

    # Input layer
    model.add(
        keras.layers.LSTM(
            lstm_hidden_units,
            input_shape=(X_shape[1], X_shape[2]),
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
    model.add(keras.layers.Dense(dense_hidden_units, activation='relu'))

    # Batch norm
    model.add(keras.layers.BatchNormalization())

    # Dropout
    model.add(keras.layers.Dropout(0.3,))

    # Output
    model.add(keras.layers.Dense(
        y_shape[2], activation='sigmoid', name='dense_output'))

    # Reshape the output
    model.add(keras.layers.Reshape(
        target_shape=(y_shape[1], y_shape[2]), name='reshaped_output'))

    # Compile it
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    # Save visuals
    keras.utils.plot_model(
        model, to_file=f'./figures/{now}_lstm.png', expand_nested=True, show_shapes=True)
    with open(f'./figures/{now}_lstm_modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

    # Save this build
    model.save(f'./untrained_models_h5/{now}_lstm.h5')


if __name__ == '__main__':
    main()
