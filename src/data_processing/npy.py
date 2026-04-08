import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_npy(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    # Load the array
    data = np.load(filepath)
    print(f"Loaded file: {os.path.basename(filepath)}")
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Min: {data.min():.4f}, Max: {data.max():.4f}, Mean: {data.mean():.4f}")

    # Ensure 2D for plotting (channels, time)
    plot_data = data.copy()
    if plot_data.ndim == 1:
        plot_data = plot_data.reshape(1, -1)
    elif plot_data.ndim > 2:
        print("Data is >2D. Slicing the first index for visualization...")
        while plot_data.ndim > 2:
            plot_data = plot_data[0]

    channels, time_steps = plot_data.shape
    # Usually we assume more time steps than channels, transpose if inverted
    if time_steps < channels:
        plot_data = plot_data.T
        channels, time_steps = plot_data.shape
        
    print(f"Plotting {channels} channels and {time_steps} time steps.")

    fig, ax = plt.subplots(figsize=(15, max(4, channels * 0.4)))
    
    # Calculate offset so channels don't heavily overlap
    # use 1.5 * standard deviation or max abs
    std = np.std(plot_data)
    if std == 0 or np.isnan(std):
        offset = 1.0
    else:
        offset = std * 4

    for i in range(channels):
        ax.plot(plot_data[i, :] + i * offset, label=f"Ch {i}", linewidth=0.8)

    ax.set_yticks(np.arange(channels) * offset)
    ax.set_yticklabels([f"Ch {i}" for i in range(channels)])
    
    ax.set_title(f"{os.path.basename(filepath)}\nOriginal Shape: {data.shape}")
    ax.set_xlabel("Samples")
    
    # Hide top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_img = "npy_visualization.png"
    plt.savefig(output_img, dpi=200, bbox_inches='tight')
    print(f"Saved visualization to {output_img}")
    plt.show()

if __name__ == "__main__":
    file_to_load = "/Volumes/WORKSPACE/opensource-dataset/Temporal-IED/DA00100S_246000_248000_500__3.npy"
    visualize_npy(file_to_load)
