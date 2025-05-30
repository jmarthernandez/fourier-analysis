import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

# Time vector
t = np.arange(0.0, 1.0, 0.001)

wave_params = []
slider_height = 0.03
slider_padding = 0.01
title_height = 0.025
slider_group_height = title_height + 3 * (slider_height + slider_padding)
slider_start_y = 0.92
next_slider_y = slider_start_y

# Set up figure with 3 axes (time, magnitude FFT, real FFT)
fig = plt.figure(figsize=(12, 8))
ax_time = fig.add_axes([0.08, 0.68, 0.6, 0.25])
ax_freq_mag = fig.add_axes([0.08, 0.38, 0.6, 0.25])
ax_freq_real = fig.add_axes([0.08, 0.08, 0.6, 0.25])

# Line plots
time_line, = ax_time.plot([], [], lw=2)
freq_mag_line, = ax_freq_mag.plot([], [], lw=2)
freq_real_line, = ax_freq_real.plot([], [], lw=2)

# Labels
ax_time.set_title("Combined Sine Wave (Time Domain)")
ax_time.set_ylabel("Amplitude")
ax_time.set_xlim(0, 1)

ax_freq_mag.set_title("FFT Magnitude (Frequency Domain)")
ax_freq_mag.set_ylabel("Magnitude")

ax_freq_real.set_title("FFT Real Part (Frequency Domain)")
ax_freq_real.set_xlabel("Frequency (Hz)")
ax_freq_real.set_ylabel("Real Value")

# FFT frequency axis
fft_freqs = np.fft.fftfreq(len(t), d=t[1] - t[0])
pos_mask = fft_freqs >= 0
fft_freqs = fft_freqs[pos_mask]

# Update function
def update(val=None):
    y = np.zeros_like(t)
    for wp in wave_params:
        amp = wp['amp_slider'].val
        freq = wp['freq_slider'].val
        phase = wp['phase_slider'].val
        y += amp * np.sin(2 * np.pi * freq * t + phase)

    time_line.set_data(t, y)
    ax_time.set_ylim(np.min(y) - 0.1, np.max(y) + 0.1 if np.max(y) != np.min(y) else 1)

    fft_vals = np.fft.fft(y)
    fft_magnitude = np.abs(fft_vals)[pos_mask]
    fft_real = np.real(fft_vals)[pos_mask]

    freq_mag_line.set_data(fft_freqs, fft_magnitude)
    freq_real_line.set_data(fft_freqs, fft_real)

    for ax in [ax_freq_mag, ax_freq_real]:
        ax.set_xlim(0, max(fft_freqs))
        ax.set_ylim(0, np.max(fft_magnitude) * 1.1 if np.max(fft_magnitude) > 0 else 1)

    fig.canvas.draw_idle()

# Add new wave sliders + title
def add_wave(event=None):
    global next_slider_y

    if next_slider_y < 0.05:
        print("⚠️ No more vertical space for sliders.")
        return

    wave_idx = len(wave_params) + 1
    x_pos = 0.75
    width = 0.2

    ax_title = fig.add_axes([x_pos, next_slider_y, width, title_height])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5, f"Wave {wave_idx}", ha="center", va="center", fontsize=10, weight='bold')

    ax_amp = fig.add_axes([x_pos, next_slider_y - title_height - 0 * (slider_height + slider_padding), width, slider_height])
    ax_freq = fig.add_axes([x_pos, next_slider_y - title_height - 1 * (slider_height + slider_padding), width, slider_height])
    ax_phase = fig.add_axes([x_pos, next_slider_y - title_height - 2 * (slider_height + slider_padding), width, slider_height])

    amp_slider = Slider(ax_amp, "Amp", 0.1, 10.0, valinit=1.0, valstep=0.1, color='green')
    freq_slider = Slider(ax_freq, "Freq", 0.5, 50.0, valinit=5.0, valstep=0.5)
    phase_slider = Slider(ax_phase, "Phase", 0, 2 * np.pi, valinit=0.0, valstep=np.pi / 16, color='orange')

    for s in (amp_slider, freq_slider, phase_slider):
        s.on_changed(update)

    wave_params.append({
        'amp_slider': amp_slider,
        'freq_slider': freq_slider,
        'phase_slider': phase_slider
    })

    next_slider_y -= slider_group_height
    update()

# Initial wave
add_wave()

# Add Wave button
ax_add = fig.add_axes([0.75, 0.01, 0.2, 0.04])
button_add = Button(ax_add, 'Add Wave', hovercolor='0.975')
button_add.on_clicked(add_wave)

plt.show()
