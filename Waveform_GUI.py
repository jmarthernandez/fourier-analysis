import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal

class WaveformGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Waveform Generator")

        self.entries = []
        self.type_vars = []

        self.setup_controls()
        self.setup_plot()

    def setup_controls(self):
        self.multiplier_type = tk.StringVar(value="linear")
        self.multiplier_factor = tk.DoubleVar(value=1.0)
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.step_var = tk.DoubleVar(value=1000)
        self.dt_var = tk.DoubleVar(value=0.01)

        ttk.Label(control_frame, text="Steps").grid(row=0, column=0)
        ttk.Entry(control_frame, textvariable=self.step_var, width=10).grid(row=0, column=1)
        ttk.Label(control_frame, text="dt").grid(row=0, column=2)
        ttk.Entry(control_frame, textvariable=self.dt_var, width=10).grid(row=0, column=3)

        ttk.Label(control_frame, text="Multiplier Type").grid(row=1, column=0)
        ttk.Combobox(control_frame, textvariable=self.multiplier_type, values=["none", "linear", "quadratic", "exponential"]).grid(row=1, column=1)
        ttk.Label(control_frame, text="Factor").grid(row=1, column=2)
        ttk.Entry(control_frame, textvariable=self.multiplier_factor, width=10).grid(row=1, column=3)

        self.functions_frame = ttk.Frame(self.root)
        self.functions_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(control_frame, text="Add Sin", command=lambda: self.add_function('sin')).grid(row=0, column=4)
        ttk.Button(control_frame, text="Add Cos", command=lambda: self.add_function('cos')).grid(row=0, column=5)
        ttk.Button(control_frame, text="Plot", command=self.plot_waveform).grid(row=0, column=6)

    def add_function(self, func_type):
        row = len(self.entries)
        type_var = tk.StringVar(value=func_type)
        self.type_vars.append(type_var)

        frame = ttk.Frame(self.functions_frame)
        frame.grid(row=row, column=0, sticky='w')

        ttk.Label(frame, text=func_type).grid(row=0, column=0)
        amp = tk.DoubleVar(value=1)
        freq = tk.DoubleVar(value=1)
        phase = tk.DoubleVar(value=0)
        
        ttk.Entry(frame, textvariable=amp, width=10).grid(row=0, column=1)
        ttk.Entry(frame, textvariable=freq, width=10).grid(row=0, column=2)
        ttk.Entry(frame, textvariable=phase, width=10).grid(row=0, column=3)

        ttk.Label(frame, text="Amplitude").grid(row=1, column=1)
        ttk.Label(frame, text="Frequency").grid(row=1, column=2)
        ttk.Label(frame, text="Phase").grid(row=1, column=3)

        self.entries.append((amp, freq, phase))

    def setup_plot(self):
        self.figure, self.axes = plt.subplots(4, 1, figsize=(8, 10))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def plot_waveform(self):
        steps = int(self.step_var.get())
        dt = self.dt_var.get()
        t = np.arange(0, steps*dt, dt)

        f = np.zeros_like(t)

        for i, (amp, freq, phase) in enumerate(self.entries):
            A = amp.get()
            w = freq.get()
            phi = phase.get()
            func_type = self.type_vars[i].get()

            if func_type == 'sin':
                f += A * np.sin(2 * np.pi * w * t + phi)
            elif func_type == 'cos':
                f += A * np.cos(2 * np.pi * w * t + phi)

        # Apply multiplicative factor
        factor = self.multiplier_factor.get()
        if self.multiplier_type.get() == "linear":
            f *= factor * t
        elif self.multiplier_type.get() == "quadratic":
            f *= factor * t**2
        elif self.multiplier_type.get() == "exponential":
            f *= np.exp(factor * t)
        elif self.multiplier_type.get() == "none":
            f *= 1
        # Clear plots
        for ax in self.axes:
            ax.clear()

        # Plot original
        self.axes[0].plot(t, f, '.')
        self.axes[0].set_title("Voltage vs Time")
        self.axes[0].set_xlabel("Time (s)")
        self.axes[0].set_ylabel("Voltage")
        self.axes[0].autoscale(enable=True, axis='both', tight=True)

        # Real Fourier Transform
        real = np.fft.rfft(f, norm='forward')
        real_freq = np.fft.rfftfreq(n=steps, d=dt)
        self.axes[1].plot(real_freq, np.abs(real))
        self.axes[1].set_title("Real Fourier Transform")
        self.axes[1].set_xlabel("Frequency (Hz)")
        self.axes[1].set_ylabel("Amplitude")
        self.axes[1].autoscale(enable=True, axis='both', tight=True)

        # Complex Fourier Transform
        complex_fft = np.fft.fft(f, norm='forward')
        freq = np.fft.fftfreq(n=len(complex_fft), d=dt)
        self.axes[2].plot(freq, np.abs(complex_fft))
        self.axes[2].set_title("Complex Fourier Transform")
        self.axes[2].set_xlabel("Frequency (Hz)")
        self.axes[2].set_ylabel("Amplitude")
        self.axes[2].autoscale(enable=True, axis='both', tight=True)

        # Spectrogram
        x, y, z = signal.spectrogram(f, fs=1/dt)
        self.axes[3].pcolormesh(y, x, z, shading='gouraud')
        self.axes[3].set_title("Spectrogram")
        self.axes[3].set_xlabel("Time (s)")
        self.axes[3].set_ylabel("Frequency (Hz)")
        self.axes[3].set_xlim([y.min(), y.max()])
        self.axes[3].set_ylim([x.min(), x.max()])

        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = WaveformGUI(root)
    root.mainloop()
