"""Training of model should be conducted here."""
import tensorflow.keras as keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
import datetime
import sys
import os
import pickle
from pathlib import Path
import time


def main():
    # Train hyperparameters
    batch_size = 128
    epochs = 256

    # Determine which model to load
    # if (len(sys.argv) != 0):
    #     load_path = sys.argv[0]
    # else:
    dir_folder = r'C:\Dev\python\CS4850\TEAM2_ORGANIZATION_REPO\models\final_fantasy\lstm\untrained_models_h5'
    paths = sorted(Path(dir_folder).iterdir(), key=os.path.getmtime)
    load_path = os.path.join(dir_folder, paths[-1].__str__())

    # Load the model
    model = keras.models.load_model(load_path)

    # Load data
    data_path = r'C:\Dev\python\CS4850\TEAM2_ORGANIZATION_REPO\playing_with_data\final_fantasy\pickled'
    with open(os.path.join(data_path, 'piano_train_val_holdval_split_dict'), 'rb') as fobj:
        train_val_holdoutval_dict = pickle.load(fobj)

    # Define training and validation sets
    X_train = train_val_holdoutval_dict['X_train']
    X_validation = train_val_holdoutval_dict['X_validation']
    y_train = train_val_holdoutval_dict['y_train']
    y_validatioon = train_val_holdoutval_dict['y_validation']

    # Track time
    now = datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')

    # Callbacks
    saved_models_path = r'C:\Dev\python\CS4850\TEAM2_ORGANIZATION_REPO\models\final_fantasy\lstm\saved_models_h5'
    checkpoint = ModelCheckpoint(
        os.path.join(saved_models_path, f'{now}_best_model.h5'),
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=64)

    # Training
    start_time = time.time()
    history = model.fit(
        X_train,
        y_train,
        verbose=1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping],
        validation_split=[],
    )
    stop_time = time.time()

    # Save the the history
    history_path = r'C:\Dev\python\CS4850\TEAM2_ORGANIZATION_REPO\models\final_fantasy\lstm\history'
    with open(os.path.join(history_path, f'{now}_pickled_history'), 'wb') as fobj:
        pickle.dump(history.history, fobj)

    # Save Training time elapsed
    wall_clock_path = r'C:\Dev\python\CS4850\TEAM2_ORGANIZATION_REPO\models\final_fantasy\lstm\wallclock'
    wall_clock = stop_time - start_time
    with open(os.path.join(wall_clock_path, '{now}_training', 'w') as fobj:
        fobj.write(f'{wall_clock} time elapsed for training on {now}')
    print(wall_clock)


if __name__ == '__main__':
    main()
