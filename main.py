import numpy as np
import matplotlib.pyplot as plt

# Define the sine wave parameters
num_waves = 40          # Number of sine waves
start_freq = 50         # Starting frequency in Hz
freq_step = 5          # Frequency increment
amplitude = 1.0        # Same amplitude for all
sampling_rate = 1000
duration = 10

sin_waves = [(start_freq + i * freq_step, amplitude) for i in range(num_waves)]

def generate_sine_wave(frequency, amplitude=1.0, duration=1.0, sampling_rate=1000):
    """
    Generate a sine wave.

    Parameters:
        frequency (float): Frequency of the sine wave in Hz.
        amplitude (float): Peak amplitude of the wave.
        duration (float): Duration in seconds.
        sampling_rate (int): Samples per second.

    Returns:
        t (np.ndarray): Time values.
        y (np.ndarray): Sine wave values.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    y = amplitude * np.sin(2 * np.pi * frequency * t)
    return t, y

def generate_combined_wave(sin_waves, duration=1.0, sampling_rate=1000):
    """
    Sum multiple sine waves into one combined signal.

    Parameters:
        sin_waves (list of tuples): Each tuple is (frequency, amplitude).
        duration (float): Duration in seconds.
        sampling_rate (int): Samples per second.

    Returns:
        t (np.ndarray): Shared time axis.
        y_total (np.ndarray): Combined sine-wave signal.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    y_total = np.zeros_like(t)
    for freq, amp in sin_waves:
        y_total += amp * np.sin(2 * np.pi * freq * t)
    return t, y_total

t, y_total = generate_combined_wave(sin_waves, duration, sampling_rate=1000)
sigma = duration / 30.0

# Build the Gaussian envelope centered at duration/2
envelope = np.exp(-0.5 * ((t - duration/2) / sigma)**2)

# Apply it to your combined wave
y_total = y_total * envelope


# Prepare plot
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

# Plot 1: combined sine wave
axs[0].plot(t, y_total, label="Combined", color="black")
axs[0].set_title("Combined Sine Wave")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Amplitude")
axs[0].set_xlim(0, duration)
axs[0].set_ylim(-num_waves-0.5, num_waves+0.5)
axs[0].legend()
axs[0].grid(True)

# Plot 2
# Compute FFT
fft_result = np.fft.fft(y_total)
N = len(y_total)
fft_magnitude = np.abs(fft_result) / N  # Initial normalization

# Frequency axis
freqs = np.fft.fftfreq(N, 1 / sampling_rate)

# Set frequency axis limit based on number of waves
max_freq = start_freq + num_waves * freq_step
axs[1].plot(freqs, fft_magnitude * 2)
axs[1].set_title("Fourier Transform - Magnitude")
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel("Magnitude")
axs[1].set_xlim(0, max_freq + freq_step)  # Show just beyond highest freq
axs[1].set_ylim(0, num_waves + 0.5)  # Adjust y-axis for better visibility
axs[1].grid(True)

plt.tight_layout()
plt.show()

#periodagram
