# Written by: Thom de Hoog
# Last edited: 2023-01-16
# This script is used to analyze the data from cells surface areas measurements over time
# Lab of Damian Brunner, University of Zurich

# repository name: cell_dynamics_analysis
# repository description: Analysis of cell dynamics data over time

#%% Importing necessary libraries
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import tifffile as tiff
import matplotlib.pyplot as plt
import tifffile as tiff
import pandas as pd
from scipy.signal import butter, filtfilt
import waipy

#%% Defining functions

def load_tiff(suffix):
    """ 
    Opens a file dialog for selecting a TIFF file, sets up the output path, and loads the image.
    Returns the loaded TIFF image and the output path for saving the processed image.
    """

    def select_file():
        """ Open a dialog to select a file and return the file path. """
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        file_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tiff *.tif")])
        return file_path

    def check_folder_exists(folder):
        """ Check if the folder exists, if not, create it. """
        if not os.path.exists(folder):
            os.makedirs(folder)
        return folder

    # Select the TIFF file
    file_path = select_file()
    if not file_path:  # Exit if no file is selected
        return None, None

    folder_path, file_name = os.path.split(file_path)

    # Define and create the output folder
    output_folder = os.path.join(folder_path, 'output')
    check_folder_exists(output_folder)

    # Define the output file base name and path
    output_file_base_name = os.path.splitext(file_name)[0] + suffix
    output_path = os.path.join(output_folder, output_file_base_name)

    # Load the TIFF file
    tiff_image = tiff.imread(file_path)

    return tiff_image, output_path


def renumber_labels(image_array):
    """
    Renumber the labels in a multi-page TIFF image array to be consecutive, starting from 1.
    Assumes the background label is always 0.

    Parameters:
    image_array (numpy.ndarray): The loaded multi-page TIFF image array.

    Returns:
    numpy.ndarray: The image array with renumbered labels.
    """

    # Initialize an array for the modified images
    modified_image_array = np.zeros_like(image_array)

    # Iterate through each page in the image array
    for i in range(image_array.shape[0]):
        img_page = image_array[i]
        img_page_modified = img_page.copy()

        # Find unique labels, excluding the background (0)
        unique_labels = np.unique(img_page[img_page != 0])

        # Create a mapping for consecutive renumbering, starting from 1
        label_mapping = {label: new_label + 1 for new_label, label in enumerate(unique_labels)}

        # Apply the renumbering
        for original_label, new_label in label_mapping.items():
            img_page_modified[img_page == original_label] = new_label

        modified_image_array[i] = img_page_modified

    return modified_image_array

# Function to calculate the area of labeled regions in each frame
def calculate_labeled_areas(frames):

    # Renumbering labels to be consecutive
    frames = renumber_labels(frames)

    # Initial setup
    num_frames = len(frames)
    max_label = np.max(frames)
    areas = {label: [] for label in range(1, max_label + 1)}

    # Calculating areas
    for frame in frames:
        labels, counts = np.unique(frame, return_counts=True)
        label_counts = dict(zip(labels, counts))
        for label in areas:
            areas[label].append(label_counts.get(label, 0))
    return areas

# Function for setting up a Butterworth bandpass filter
def butterworth_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to apply the bandpass filter to data
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butterworth_bandpass(lowcut, highcut, 1/fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Function to plot filtered data for all objects
def plot_filtered_data_all_objects(filtered_areas, output_path):
    
    # Plotting settings
    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap('tab20', len(filtered_areas))
    
    # Plotting filtered data for each object
    for idx, (label, areas) in enumerate(filtered_areas.items()):
        plt.plot(np.arange(len(areas)) * 30, areas, label=f'Object {label}', color=colors(idx))
    
    # Plotting settings
    plt.xlabel('Time (seconds)')
    plt.ylabel('Filtered Area')
    plt.title('Filtered Area of Labeled Objects Over Time')
    plt.legend()
    plt.grid(True)

    # Saving plot
    plot_filename_png = output_path + '_filtered.png'
    plot_filename_pdf = output_path+ '_filtered.pdf'
    plt.savefig(plot_filename_png, format='png')
    plt.savefig(plot_filename_pdf, format='pdf')

    plt.show()

# Function to save filtered areas to a CSV file
def write_csv(unfiltered_areas, filtered_areas, fs, output_path, suffix):
    # Creating DataFrame from unfiltered areas and renaming columns
    unfiltered_df = pd.DataFrame(unfiltered_areas)
    unfiltered_df.columns = ['Raw_Area_' + str(col) for col in unfiltered_df.columns]

    # Creating DataFrame from filtered areas and renaming columns
    filtered_df = pd.DataFrame(filtered_areas)
    filtered_df.columns = ['Filtered_Area_' + str(col) for col in filtered_df.columns]

    # Combining both DataFrames horizontally
    combined_df = pd.concat([unfiltered_df, filtered_df], axis=1)

    # Creating a time column (in seconds)
    time_column = np.arange(0, len(combined_df) * fs, fs)
    combined_df.insert(0, 'sec', time_column)

    # Saving to CSV
    csv_filename = output_path + suffix + '.csv'
    combined_df.to_csv(csv_filename, index=False)

    return combined_df

# Function to write period data to CSV
def write_period_data_to_csv(period_cwt, output_path):
    period_data = pd.DataFrame({'Most_common_period_in_sec': period_cwt})
    csv_filename = output_path + '_period_data.csv'
    period_data.to_csv(csv_filename)

def cwtAndFFT(data, fs, label, folder_path):

    # Calculating period using continuous wavelet transform and fast fourier transform

    # Wavelet analysis specifics from waipy
    dt = fs                                 # Define the sampling interval
    pad = 1                                 # pad the time series with zeroes (recommended)
    dj = 1/256                              # this will do 256 sub-octaves per octaves
    s0 = 2*dt                               # this says start at a where the scale starts
    j1 = 7/dj                               # this says do 7 powers-of-two with dj sub-octaves each
    mother = 'Morlet'                       # Define the wavelet mother function
    param = 6                               # this is the order of the mother wavelet for the wavelet transform 
    dtmin = 1/10000000000                   # minimum spacing between peaks, ideally zero. Smaller number will find more peaks
    t = np.arange(0, len(data) * dt, dt)    # construct time array

    # Normalizing data and calculating alpha value, which is the lag-1 autocorrelation
    data_norm = waipy.normalize(data)
    alpha = np.corrcoef(data_norm[0:-1], data_norm[1:])[0,1]

    # Performing the continuous wavelet transform
    result = waipy.cwt(data_norm, dt, pad, dj, s0, j1, alpha, param, mother,name='x')
    
    # Plotting continuous wavelet transform
    waipy.wavelet_plot('Wavelet analysis cell area ' + str(label), t, data_norm, dtmin, result, label= 'Cell ' + str(label), cmap='turbo')

    # Saving plot of continuous wavelet transform
    plot_filename_pdf = folder_path + str(label) + '_cwt-cell' + str(label) + '.pdf'
    plt.savefig(plot_filename_pdf, format='pdf')

    # Taking the absolute value of the wavelet transform
    # and calculating extracting the period
    index_ws = np.argmax(result['global_ws'])
    get_most_common_period_cwt = result['period'][index_ws]


    return get_most_common_period_cwt

#%% Load the .tiff file and set up variables
tiff_image, output_path = load_tiff()

# Setting up variables
sec_per_frame= 30
bandpass_lowcut = 1/600
bandpass_highcut = 1/120

#%% Calculating labeled areas over time
labeled_areas_over_time_new = calculate_labeled_areas(tiff_image)

#%% Applying the butterworth bandpass filter to the data
filtered_areas = {label: bandpass_filter(areas, bandpass_lowcut, bandpass_highcut, sec_per_frame) 
                  for label, areas in labeled_areas_over_time_new.items()}

#%% Plotting filtered data for all objects
plot_filtered_data_all_objects(filtered_areas, output_path)

#%% Applying the Continuous Wavelet Transform (CWT) to get the most common period in the data
period_from_cwt = {}

# Plotting filtered data for all objects
for label in filtered_areas:
    cwt_result = cwtAndFFT(filtered_areas[label], sec_per_frame, label, output_path)
    period_from_cwt[label] = cwt_result

#%% Save the data and period data to a csv file
write_csv(labeled_areas_over_time_new, filtered_areas, sec_per_frame, output_path, '_filtered_areas')
write_period_data_to_csv(period_from_cwt, output_path)
