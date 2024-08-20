import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_important_features(validation_folder_path, file_names, features_to_plot):
    fig, axs = plt.subplots(len(features_to_plot), len(file_names), figsize=(15, 12))
    fig.suptitle('Comparison of Selected Features Across Validation Files')

    colors = ['b', 'r', 'g', 'm', 'c'] 

    for j, file_name in enumerate(file_names):
        # Load validation data
        validation_data_path = os.path.join(validation_folder_path, f"{file_name}.csv")
        validation_data = pd.read_csv(validation_data_path)

        for i, feature in enumerate(features_to_plot):
            # Check if there are multiple files to plot
            if len(file_names) > 1:
                ax = axs[i, j]
            else:
                ax = axs[i]

            # Plotting for the current file
            ax.plot(validation_data['time'], validation_data[feature], label=f'{feature} in {file_name}', color=colors[j])
            ax.set_title(f'{feature} in {file_name}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

validation_folder_path = 'C:/Users/andry/Desktop/492/midterm/validation/'
file_names = ['1', '2']  #ive edited this for all the different comparisons
features_to_plot = ['prd_s', 'lq_s', 'prd_c', 'prd_a']  #ive edited this for all the different comparisons

plot_important_features(validation_folder_path, file_names, features_to_plot)

