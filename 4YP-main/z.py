import matplotlib.pyplot as plt
import numpy as np

# Epoch numbers (1 to 47)
# epochs = np.arange(1, 48# 




# 
epochs = np.arange(1, 46)

# Loss values from each epoch
loss = np.array([
    -0.7158, -0.8089, -0.8084, -0.8563, -0.8824, -0.9010, -0.9180, -0.9410, -0.9951,
    -1.0099, -0.9894, -1.0095, -1.0342, -1.0283, -1.0755, -1.0082, -1.0419, -1.0204,
    -1.0307, -1.0523, -1.0477, -1.0797, -1.0898, -1.0867, -1.1140, -1.0857, -1.0244,
    -1.0767, -1.0965, -1.0361, -1.0832, -1.0632, -1.0738, -1.0803, -1.0899, -1.1246,
    -1.1002, -1.1040, -1.1026, -1.1203, -1.1195, -1.1386, -1.1310, -1.0841, -1.1237,
    # -1.1052, -1.0831
])

# Validation loss values from each epoch
val_loss = np.array([
    -0.8431, -0.8363, -0.8621, -0.8189, -0.7665, -0.8223, -0.7929, -0.8445, -0.7722,
    -0.7924, -0.7652, -0.7843, -0.8162, -0.8903, -0.8415, -0.8552, -0.8980, -0.8803,
    -0.7838, -0.9096, -0.8234, -0.9400, -0.9172, -0.7147, -0.8378, -0.8454, -0.8661,
    -0.9007, -0.6525, -0.8192, -0.7767, -0.7936, -0.8100, -0.9078, -0.9015, -0.9038,
    -0.8750, -0.8658, -0.8823, -0.8858, -0.8654, -0.9043, -0.9044, -0.8694, -0.8646,
    # -0.7697, -0.8611
])

def moving_average(data, window_size=3):
    """Compute the moving average with the given window size."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Calculate moving averages
loss_ma = moving_average(loss, window_size=3)
val_loss_ma = moving_average(val_loss, window_size=3)

# For the moving average plot, adjust epoch numbers to match the averaged values
epochs_ma = np.arange(1 + (3 - 1) // 2, len(loss) - (3 - 1) // 2 + 1)

# Create a figure with a specific size
plt.figure(figsize=(10, 6))

# Plot original loss values and moving average
plt.plot(epochs, loss, label='Loss', marker='o', linestyle='--', alpha=0.5)
plt.plot(epochs_ma, loss_ma, label='Loss MA (window=3)', marker='o')

# Plot original validation loss values and moving average
plt.plot(epochs, val_loss, label='Validation Loss', marker='x', linestyle='--', alpha=0.5)
plt.plot(epochs_ma, val_loss_ma, label='Val Loss MA (window=3)', marker='x')

# Labeling the axes and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss vs Epochs with Moving Averages')

# Display the legend and grid
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
