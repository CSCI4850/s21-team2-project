"""GUI for downloading VGMusic MIDI files.

A control variable of <class 'tkinter.BooleanVar'> controls the threaded
execution of the 'scrape_midis' function. Currently, the data is 
downloaded to the current directory of the exe, zipped, and then
saved or discarded based on the user input. Using 'tempfile' stdlibrary 
may be a better choice, but the current strategy will suffice.

References:

-- pyinstaller
https://stackoverflow.com/questions/17584698/getting-rid-of-console-output-when-freezing-python-programs-using-pyinstaller

-- Threading
https://www.geeksforgeeks.org/how-to-use-thread-in-tkinter-python/

-- General 
https://coderslegacy.com/python/python-gui/

-- jfdev001
nmr-inte-great
tkinter-calculator
"""

# Imports
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import shutil
import os
from pathlib import Path
import threading
import time
import random
import requests
import bs4


# The directory in which scraped files will initially be stored
TMP_FILE_DIRECTORY = os.getcwd()


##################
# Helper functions
##################

def scrape():
    """Scrape all MIDI files for a videogame and store on disk."""
    # Scrape the list of midis to be downloaded
    midis = list_of_vg_midis(entry_var.get())

    # Check if midis were scraped
    if (midis is None):
        return messagebox.showerror(title='Error',
                                    message='Videogame Not Found')

    # Get user response
    response = messagebox.askquestion(
        title='Update', message='The game exists. Still want to scrape its MIDIs?')

    # Check user response
    if (response == 'yes'):

        # Scrape the data using threading
        t1 = threading.Thread(target=scrape_midis, args=(midis,))
        t1.start()

    else:

        # Alert user of the canceled scraping
        messagebox.showinfo(title='Update', message='Scraping Canceled')

    # Void function
    return None


def interrupt_scraper():
    """Interrupts the scraper by updating the control variable."""
    # Update the scraper control variable
    stop_var.set(True)

    # Enable the <class 'tkinter.Entry'> widget again
    entry_name_of_vg.config(state='normal')

    # Void function
    return None


def scrape_midis(midis, write_to_dir=TMP_FILE_DIRECTORY):
    """Downloads the MIDIs for a game and zips them up

    :param midis: <class 'list'> of midi names to scrape from VGMusic.
    :param write_to_dir: <class 'str'> The folder to write the data to.
        Defaults to 'cwd/data/'
    :return: None
    """
    # Prevent the user from changing the <class 'tkinter.Entry'> widget
    entry_name_of_vg.config(state='disabled')

    # URL to scrape
    URL = "https://www.vgmusic.com/music/console/nintendo/nes/"

    # Dir path
    dir_path = os.path.join(write_to_dir, entry_var.get())

    # Check if the write directory exists and create it if not
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Iterate through midis and write to disk
    for ix, midi in enumerate(midis):

        # Check the control variable <class 'tkinter.BooleanVar'>
        if (stop_var.get()):

            # Delete files if they exist
            if (os.path.exists(dir_path)):
                shutil.rmtree(dir_path)

            # Reset the stop var
            stop_var.set(False)

            # Reset the counter vars
            count_var.set(0)
            total_songs_var.set(0)
            progress_var.set(
                f'Progress: {str(count_var.get())}/{str(total_songs_var.get())}')

            # Enable the <class 'tkinter.Entry'> widget again
            entry_name_of_vg.config(state='normal')

            # Alert user of interrupted process
            return messagebox.showerror(title='Interrupt',
                                        message='Scraping Interrupted')
        else:
            # Download the data
            time.sleep(random.uniform(1, 5))
            data_res = requests.get(
                URL + '/' + midi, allow_redirects=True).content

            # Write the data
            with open(os.path.join(dir_path, midi), "wb") as fobj:
                fobj.write(data_res)

            # Update counter vars
            count_var.set(ix + 1)
            progress_var.set(
                f'Progress: {str(count_var.get())}/{str(total_songs_var.get())}')

    # Zip the files and remove the previous dir path where files
    # were temporarily stored
    shutil.make_archive(entry_var.get(), 'zip', dir_path)
    shutil.rmtree(dir_path)

    # Ask user if they want to save the zip file
    want_to_save = messagebox.askquestion(
        title='To Save or Not To Save',
        message='Do you want to save the MIDIs you just downloaded?')

    # Check response
    if (want_to_save == 'yes'):

        # Returns directory name into which plot will be saved
        save_dir = filedialog.askdirectory(title="Select Directory",
                                           initialdir=str(Path.home()))

        # Move the zip file in the current directory to the desired
        # director
        try:
            shutil.move(os.path.join(os.getcwd(), entry_var.get() + '.zip'),
                        save_dir)
        except:
            messagebox.showerror(title='Error',
                                 message='Problem Saving the File. Make Sure It Does Not Already Exist.')
            os.remove(os.path.join(os.getcwd(), entry_var.get() + '.zip'))

    else:

        # Delete the zip file
        os.remove(os.path.join(os.getcwd(), entry_var.get() + '.zip'))

    # Reset the stop_var
    stop_var.set(False)

    # Reset the counter vars
    count_var.set(0)
    total_songs_var.set(0)
    progress_var.set(
        f'Progress: {str(count_var.get())}/{str(total_songs_var.get())}')

    # Enable the <class 'tkinter.Entry'> widget again
    entry_name_of_vg.config(state='normal')

    # Void function
    return None


def list_of_vg_midis(name_of_vg):
    """Requests site and gets list of midi files for a videogame.

    :param name_of_vg: <class 'str'> The name of a videogame
        from which the list of midis will be created.
    :return: <class 'list'> of all .mid files for a particular 
        videogame.
    """
    # URL to scrape
    URL = "https://www.vgmusic.com/music/console/nintendo/nes/"

    # Get the HTTP response content for this URL
    res = requests.get(URL).content

    # Get the soup obj
    soup = bs4.BeautifulSoup(res, "html.parser")

    # Find where the name of the videogame is in the bs4 tree
    vg_str = soup.find(string=name_of_vg)

    # Exit the function and return nothing
    if (vg_str is None):
        return None

    # Get the parent tag of that videogame as a starting point -- can go straight to find next if necessary
    vg_tr_header = vg_str.find_parent('tr')

    # Get the next tag after that table header. This
    # represents an html row containing the midi, file size, who sequenced it, and comments
    tr_tag = vg_tr_header.find_next('tr')

    # Initialize loop var. When an html row is whitespace ONLY, the
    # videogame has no more midi files associated with it.
    is_whitespace = tr_tag.get_text().isspace()
    midis = []
    while(not is_whitespace):
        # Get the midi
        midi = tr_tag.find('a').get('href')
        midis.append(midi)

        # Get the next tag
        tr_tag = tr_tag.find_next('tr')

        # Update loop var
        is_whitespace = tr_tag.get_text().isspace()

    # Set the total number of midis var
    total_songs_var.set(len(midis))

    # Set the progress variable for the label
    progress_var.set(
        f'Progress: {str(count_var.get())}/{str(total_songs_var.get())}')

    # Return the list of midis
    return midis


##################
# The tkinter GUI
##################

# Root window
root = tk.Tk()
root.title('VGMScraper')
root.geometry('400x150')
root.resizable(0, 0)

# The frame within root
frame = tk.Frame(root)
frame.pack()

# <class 'tkinter.Entry'> for the name of video game
entry_var = tk.StringVar(
    frame, value=' Videogame Name to Download')
entry_name_of_vg = tk.Entry(
    frame, textvariable=entry_var, width=50, relief=tk.SUNKEN)
entry_name_of_vg.pack(padx=5, pady=5)

# <class 'tkinter.Button'> to scrape the data
button_scrape = tk.Button(frame, text='Click to Scrape',
                          command=scrape, width=25)
button_scrape.pack(padx=5, pady=5)

# <class 'tkinter.Button'> to stop the scraping of data
stop_var = tk.BooleanVar(frame, value=False)
button_stop_scrape = tk.Button(
    frame, text='CLICK TO INTERRUPT', command=interrupt_scraper, width=25,
    bg='tomato')
button_stop_scrape.pack(padx=5, pady=5)

# <class 'tkinter.Label'> for displaying how many files are left
count_var = tk.IntVar(frame, value=0)
total_songs_var = tk.IntVar(frame, value=0)
progress_var = tk.StringVar(
    frame, value=f'Progress: {str(count_var.get())}/{str(total_songs_var.get())}')
label_track_progress = tk.Label(frame,
                                textvariable=progress_var,
                                relief=tk.RAISED,
                                width=25)
label_track_progress.pack(padx=5, pady=5)

# Main Loop
root.mainloop()
