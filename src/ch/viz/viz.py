# Visualization utilities for plots and figures.

import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap(matrix, title=None, xlabel=None, ylabel=None, cmap='viridis', colorbar=True, save_path=None):
    """
    Plots a heatmap of a given matrix.

    Parameters:
        matrix (np.ndarray): 2D array to visualize as a heatmap.
        title (str, optional): Title of the heatmap.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        cmap (str, optional): Colormap to use for the heatmap. Default is 'viridis'.
        colorbar (bool, optional): Whether to include a colorbar. Default is True.
        save_path (str, optional): Path to save the heatmap image. If None, the plot is shown instead.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap=cmap, aspect='auto')
    
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    if colorbar:
        plt.colorbar()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()

    plt.close()

