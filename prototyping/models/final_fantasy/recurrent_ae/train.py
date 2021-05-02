"""Training of model should be conducted here."""
import tensorflow.keras as keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
import datetime
import sys
import os
import pickle
from pathlib import Path
import time
from constants import PICKLE_PATH, INPUT_TIMESTEPS, OUTPUT_TIMESTEPS


def main():
    # Train hyperparameters
    batch_size = 128
    epochs = 128

    # Load most recently created model
    cwd = os.path.dirname(os.path.realpath(__file__))
    paths = sorted(
        Path(
            os.path.join(
                cwd, './untrained_models_h5'
            )
        ).iterdir(), key=os.path.getmtime)
    load_path = paths[-1].__str__()

    # Load the model
    model = keras.models.load_model(load_path)

    # Load data
    with open(os.path.join(PICKLE_PATH, f'piano_train_val_holdval_split_for_time_series_LABELED_{INPUT_TIMESTEPS}_input_{OUTPUT_TIMESTEPS}_output_dict'), 'rb') as fobj:
        train_val_holdoutval_dict = pickle.load(fobj)

    # Define training and validation sets
    X_train = train_val_holdoutval_dict['X_train']
    X_validation = train_val_holdoutval_dict['X_validation']
    y_train = train_val_holdoutval_dict['y_train']
    y_validation = train_val_holdoutval_dict['y_validation']

    # Track time
    now = datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')

    # Callbacks for validation sets
    # Based on the metrics used for model build -- see experiments.xlsx
    checkpoint_val_loss = ModelCheckpoint(
        os.path.join(cwd, 'saved_models_h5', f'{now}_val_loss_model.h5'),
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1)
    checkpoint_val_prec = ModelCheckpoint(
        os.path.join(cwd, 'saved_models_h5', f'{now}_val_precision_model.h5'),
        monitor='val_precision',
        mode='max',
        save_best_only=True,
        verbose=1)
    checkpoint_val_recall = ModelCheckpoint(
        os.path.join(cwd, 'saved_models_h5', f'{now}_val_recall_model.h5'),
        monitor='val_recall',
        mode='max',
        save_best_only=True,
        verbose=1)
    checkpoint_val_kld = ModelCheckpoint(
        os.path.join(cwd, 'saved_models_h5',
                     f'{now}_val_kullback_leibler_divergence_model.h5'),
        monitor='val_kullback_leibler_divergence',
        mode='min',
        save_best_only=True,
        verbose=1)

    #early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=64)

    # Training
    start_time = time.time()
    history = model.fit(
        X_train,
        y_train,
        verbose=1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_val_loss,
                   checkpoint_val_prec, checkpoint_val_recall, checkpoint_val_kld],
        validation_data=[X_validation, y_validation],
    )
    stop_time = time.time()

    # LOG HISTORY KEYS
    print('History keys:', history.history)

    # Save the the history
    with open(os.path.join(cwd, 'history', f'{now}_history'), 'wb') as fobj:
        pickle.dump(history.history.keys(), fobj)

    # Save Training time elapsed
    wall_clock = stop_time - start_time
    with open(os.path.join(cwd, 'wallclock', f'{now}_training'), 'w') as fobj:
        fobj.write(f'{wall_clock} time elapsed for training on {now}')

    # LOG
    print(f'Training completed in {wall_clock} seconds.')


if __name__ == '__main__':
    main()
