#%%
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats

# Functions
def select_and_load_csv(suffix):
    """
    Opens a file dialog for selecting a CSV file and loads it.
    Returns the loaded DataFrame or None if no file is selected.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    csv_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    
    if not csv_file_path:  # Exit if no file is selected
        print("No CSV file selected.")
        return None, None, None

    try:
        df = pd.read_csv(csv_file_path)
        df = df.dropna()  # Drop rows with any NaN values
        print("CSV file loaded successfully.")

        # Get the folder path and file name
        folder_path, file_name = os.path.split(csv_file_path)

        # Define and create the output folder
        output_folder = os.path.join(folder_path, 'outputplot')
        check_folder_exists(output_folder)

        # Define the output file base name and path
        output_file_base_name = os.path.splitext(file_name)[0] + suffix
        output_path = os.path.join(output_folder, output_file_base_name)

        return df, csv_file_path, output_path
    
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
        return None, None, None

def check_folder_exists(folder):
    """ Check if the folder exists, if not, create it. """
    if not os.path.exists(folder):
        os.makedirs(folder)
        return folder

# Load the CSV file
df, csv_file_path, output_path = select_and_load_csv('_plot')

if df is not None:

    # Define conditions
    condition1 = 'unclustered'
    condition2 = 'clustered'

    # Calculate means and standard errors for error bars
    means = df.groupby('condition')['data'].mean()
    std_errors = df.groupby('condition')['data'].sem()

    # Calculate sample sizes for each condition
    n_condition1 = df[df['condition'] == condition1].shape[0]
    n_condition2 = df[df['condition'] == condition2].shape[0]

    # Perform Mann-Whitney U test
    _, p_value = stats.mannwhitneyu(
        df[df['condition'] == condition1]['data'],
        df[df['condition'] == condition2]['data'],
        alternative='two-sided'
    )

    # Determine significance level
    if p_value < 0.0001:
        significance = '****'
    elif p_value < 0.001:
        significance = '***'
    elif p_value < 0.01:
        significance = '**'
    elif p_value < 0.05:
        significance = '*'
    else:
        significance = 'n.s.'

    # Colors for each condition
    colors_condition1 = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    colors_condition2 = ['#ff7f0e', '#8c564b', '#e377c2', '#bcbd22']
    colors_combined = colors_condition1 + colors_condition2

    # Creating a color mapping for embryos
    embryos = df['embryo'].unique()
    embryo_shade_mapping = {embryo: color for embryo, color in zip(embryos, colors_combined)}

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))

    # Swarmplot with color palette for embryos
    sns.swarmplot(
        x='condition', y='data', hue='embryo',
        palette=embryo_shade_mapping, data=df, ax=ax,
        marker="o", size=12, edgecolor=None, zorder=0
    )

    # Error bars with whiskers
    conditions = df['condition'].unique()
    for i, condition in enumerate(conditions):
        ax.errorbar(
            x=i, y=means[condition], yerr=std_errors[condition],
            color='black', capsize=5, fmt='none', lw=2.5,
            zorder=1, capthick=2.5
        )

    # Determine the current range of the y-axis
    y_min, y_max = df['data'].min(), df['data'].max()
    y_range = y_max - y_min # Calculate the range
    y_range_expanded = y_range * 0.2 # Increase the range by 10%
    ax.set_ylim(y_min - y_range_expanded, y_max + y_range_expanded)     # Set new y-axis limits

    # Determine the range of the y-axis
    y_min, y_max = df['data'].min(), df['data'].max()
    y_range = y_max - y_min

    # Calculate proportional offsets
    offset = y_range * 0.05  # Adjust this factor as needed
    y_bracket_height = y_max + offset
    y_p_value_height = y_bracket_height + offset / 2  # Half of the original offset

    # Adding the statistical significance bracket and p-value
    ax.plot([0, 0, 1, 1], [y_bracket_height, y_p_value_height, y_p_value_height, y_bracket_height], color='black', lw=1.5)
    ax.text(0.5, y_p_value_height + offset / 2, significance, ha='center', va='center', fontsize=12)

    # Updated plot aesthetics
    ax.set_xlim(-0.7, 1.7)
    ax.grid(False)
    ax.tick_params(axis='both', which='both', length=5)
    ax.set_title('Comparison of Conditions')
    ax.set_ylabel('Data Value')
    ax.set_xticklabels([f'{condition} (N={n})' for condition, n in zip(conditions, [n_condition1, n_condition2])])

    # Ensure embryos is a list
    embryos_list = list(df['embryo'].unique())
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:len(embryos_list)], labels=embryos_list, loc='upper left', bbox_to_anchor=(1.05, 1), title='Embryo')

    #plt.tight_layout()
    plt.show()

    # Saving the plot as PDF
    fig.savefig(output_path + '.pdf', format='pdf', bbox_inches='tight')
else:
    print("Data loading was aborted.")

# %%
