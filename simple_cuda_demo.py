import torch
import numpy as np
import time

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'CUDA is available. Using device: {torch.cuda.get_device_name(0)}')
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU.')

# Create a synthetic ECG-like signal (1000 samples, 1 lead)
fs = 100  # Hz
t = np.linspace(0, 10, 10*fs)  # 10 seconds
# Simple ECG simulation: sum of sine waves with added noise
ecg = np.sin(2*np.pi*1.0*t) + 0.5*np.sin(2*np.pi*20.0*t) + 0.2*np.random.randn(len(t))
ecg = ecg.astype(np.float32)

print(f'Signal shape: {ecg.shape}')

# Move to device
ecg_tensor = torch.from_numpy(ecg).to(device)

# Define a simple low-pass filter (Moving average) using convolution on GPU
def moving_average_filter(signal, kernel_size=5):
    # Create kernel
    kernel = torch.ones(kernel_size, device=device) / kernel_size
    # Pad signal to maintain size
    padding = kernel_size // 2
    padded = torch.nn.functional.pad(signal.unsqueeze(0).unsqueeze(0), (padding, padding), mode='reflect')
    # Convolve
    filtered = torch.nn.functional.conv1d(padded, kernel.unsqueeze(0).unsqueeze(0))
    return filtered.squeeze()

# Apply filter
start = time.time()
ecg_filtered = moving_average_filter(ecg_tensor, kernel_size=11)
torch.cuda.synchronize() if device.type == 'cuda' else None  # Wait for GPU to finish
end = time.time()

print(f'Filtering took {end-start:.4f} seconds on {device}.')

# Move result back to CPU for inspection (if needed)
ecg_filtered_np = ecg_filtered.cpu().numpy()

# Show first 10 samples
print('First 10 samples of original signal:', ecg[:10])
print('First 10 samples of filtered signal:', ecg_filtered_np[:10])

# Simple verification: check that the filtered signal is indeed smoothed (variance lower)
print(f'Original signal variance: {np.var(ecg):.6f}')
print(f'Filtered signal variance: {np.var(ecg_filtered_np):.6f}')

print('Demo completed.')