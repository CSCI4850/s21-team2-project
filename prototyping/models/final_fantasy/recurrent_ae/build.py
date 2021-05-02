"""Construct LSTM-AE."""

# Imports
import pickle
import os
import tensorflow.keras as keras
import datetime
from contextlib import redirect_stdout
import pickle
from constants import PICKLE_PATH, INPUT_TIMESTEPS, OUTPUT_TIMESTEPS


def main():
    # Build hyperparameters
    lstm_base_units = 256
    lstm_hidden_units = 128

    # Path to pickle data
    cwd = os.path.dirname(os.path.realpath(__file__))

    # Load the labeled input output matrix data
    data_path = os.path.join(
        PICKLE_PATH, f'piano_train_val_holdval_split_for_time_series_LABELED_{INPUT_TIMESTEPS}_input_{OUTPUT_TIMESTEPS}_output_dict')
    with open(data_path, 'rb') as fobj:
        train_test_holdoutval_dict = pickle.load(fobj)

    ###################
    # Build the model #
    ###################

    # Shape of data
    # (?, timesteps, categories)
    X_shape = train_test_holdoutval_dict['X_train'].shape
    y_shape = train_test_holdoutval_dict['y_train'].shape

    # Save current time
    now = datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')

    # Instantiate the sequential model
    name = f'{now}_lstm_ae'
    model = keras.models.Sequential(name=name)

    ####################
    # Recurrent Encoder#
    ####################
    # Input layer
    model.add(
        keras.layers.LSTM(
            lstm_base_units,
            input_shape=(X_shape[1], X_shape[2]),
            recurrent_dropout=0.3,
            return_sequences=True,
            name='input_lstm'
        ))

    # Encoder hidden
    model.add(keras.layers.LSTM(lstm_hidden_units))

    ####################
    # Recurrent Decoder#
    ####################
    # Repeat the squashed output but with the timestep dimension
    model.add(keras.layers.RepeatVector(y_shape[1]))

    # Last hidden
    model.add(keras.layers.LSTM(lstm_base_units, return_sequences=True))

    # Output
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Dense(
                y_shape[2], activation='sigmoid', name='dense_output'
            )
        )
    )

    ##############
    # Compile it #
    ##############
    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=[keras.metrics.Precision(), keras.metrics.Recall(),
                 keras.metrics.KLDivergence()]
    )

    ##############
    #   Saving   #
    ##############
    # Save visuals
    keras.utils.plot_model(
        model,
        to_file=os.path.join(cwd, f'./figures/{now}_lstm_ae.png'),
        expand_nested=True,
        show_shapes=True
    )

    with open(os.path.join(cwd, f'./figures/{now}_lstm_ae_modelsummary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()

    # Save this build
    model.save(os.path.join(cwd, f'./untrained_models_h5/{now}_lstm_ae.h5'))

    # LOG
    print(
        f'Build {now} with {INPUT_TIMESTEPS} input and {OUTPUT_TIMESTEPS} dimensions complete.')


if __name__ == '__main__':
    main()
