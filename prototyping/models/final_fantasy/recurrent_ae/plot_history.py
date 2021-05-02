"""Plot the history of a saved model."""

# Imports
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path, PurePath
import math


def plot_learning_curve(history, suptitle):
    """Display matplotlib diagnostic figures for the model-history.

    :param history: <class 'keras.callbacks.History'>
    :param suptitle: <class 'str'> to set the overall title of all
        the subplots
    :return: <class 'matplotlib.figure.Figure'>
    """

    # Determine the number of subplots based on the number of keys
    # in the history dict
    num_keys = len(history.keys())

    # List of str keys
    keys = list(history.keys())

    # Check to see if validation data, if so then you can just divide by
    # two and then take those as rows... (not good for more than 3 metrics)
    check_for_val_key = [key for key in keys if 'val' in key]
    one_dim = False
    if (len(check_for_val_key) > 0):
        n_rows = 1
        n_cols = num_keys // 2
        one_dim = True
    else:

        # Determine dimensions since simple dims weren't possible
        if (num_keys % 2 == 0):
            n_rows = num_keys // 2
            n_cols = num_keys // 2
        else:
            n_rows = int(math.ceil(num_keys / 2))
            n_cols = int(math.floor(num_keys / 2))

    # MPL initial objs
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15,  8))

    # Iterate through keys in dict and plot values vs. number of epochs
    # Number of epochs is implicit argument since y_arr is first arg
    key_ix = 0
    for row in range(n_rows):
        for col in range(n_cols):
            try:
                # Current key
                cur_key = keys[key_ix]

                # Check dims
                if (not one_dim):
                    # Set the plot title
                    axs[row, col].set_title(cur_key)

                    # Plot training data
                    axs[row, col].plot(history[cur_key], label=cur_key)
                else:
                    # Set the plot title
                    axs[col].set_title(cur_key)

                    # Plot training data
                    axs[col].plot(history[cur_key], label=cur_key)

                # Check for validation data
                if ('val_' + cur_key in keys):

                    # Get the index of that validation key
                    val_ix = keys.index('val_' + cur_key)

                    # Plot the validation data
                    if (not one_dim):
                        axs[row, col].plot(
                            history[keys[val_ix]], label=keys[val_ix])
                    else:
                        axs[col].plot(
                            history[keys[val_ix]], label=keys[val_ix])

                    # Delete the plotted validation key
                    del keys[val_ix]

                # Add a legend to the plot
                if (not one_dim):
                    axs[row, col].legend()
                else:
                    axs[col].legend()

                # Update the key_ix for iteration through the keys
                # dictionary as a list
                key_ix += 1

            except IndexError:
                axs[row, col].set_axis_off()
                print('Index out of range.')

    # Set the overall title
    fig.suptitle(suptitle)

    # Return the figure
    return fig


def main():
    # Current working directory
    cwd = os.path.dirname(os.path.realpath(__file__))

    # Sorts the files in the listed directory by their modification time
    # where the first element is the oldest and the last element of the
    # returned list is the newest. The list contains windows path
    # objects and the __str__() method converts that obj --> str
    most_recent_history_path = sorted(Path(os.path.join(cwd, './history')
                                           ).iterdir(), key=os.path.getmtime)[-1].__str__()

    # Load model history
    with open(most_recent_history_path, 'rb') as fobj:
        history = pickle.load(fobj)

    # Get the name of the file that was processed
    fname = PurePath(most_recent_history_path).name

    # Plot the learning curve of the model history
    learning_curve = plot_learning_curve(history, suptitle=fname)

    # Save the figure
    learning_curve.savefig(os.path.join(
        cwd, './figures', fname), bbox_inches='tight')

    # LOG
    print('Figures created and saved.')


if __name__ == '__main__':
    main()
