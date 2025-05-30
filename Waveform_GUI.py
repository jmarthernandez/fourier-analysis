import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy import signal
import json

class WaveformGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Waveform Generator")

        self.entries = []
        self.type_vars = []
        self.freq_change_vars = []
        self.t_naught_vars = []
        self.frames = []

        self.setup_controls()
        self.setup_plot()

    def setup_controls(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        general_frame = ttk.Frame(notebook)
        waveform_frame = ttk.Frame(notebook)
        export_frame = ttk.Frame(notebook)

        notebook.add(general_frame, text='General Settings')
        notebook.add(waveform_frame, text='Waveform Components')
        notebook.add(export_frame, text='Export Options')

        # General settings
        self.multiplier_type = tk.StringVar(value="linear")
        self.multiplier_factor = tk.DoubleVar(value=1.0)
        self.meu = tk.DoubleVar(value=0.5)
        self.sigma = tk.DoubleVar(value=0.1)

        self.step_var = tk.DoubleVar(value=1000)
        self.dt_var = tk.DoubleVar(value=0.01)

        ttk.Label(general_frame, text="Steps").grid(row=0, column=0)
        ttk.Spinbox(general_frame, from_=100, to=100000, textvariable=self.step_var, width=10).grid(row=0, column=1)
        ttk.Label(general_frame, text="dt (s)").grid(row=0, column=2)
        ttk.Entry(general_frame, textvariable=self.dt_var, width=10).grid(row=0, column=3)
        self.total_time_label = ttk.Label(general_frame, text="Total time: 0.0")
        self.total_time_label.grid(row=0, column=4)
        self.sampling_rate_label = ttk.Label(general_frame, text="Sampling rate: 0.0 Hz")
        self.sampling_rate_label.grid(row=0, column=5)

        ttk.Label(general_frame, text="Multiplier Type").grid(row=1, column=0)
        ttk.Combobox(general_frame, textvariable=self.multiplier_type,
                     values=["none", "linear", "quadratic", "exponential", "envelope"]).grid(row=1, column=1)
        ttk.Label(general_frame, text="Factor").grid(row=1, column=2)
        ttk.Entry(general_frame, textvariable=self.multiplier_factor, width=10).grid(row=1, column=3)
        ttk.Label(general_frame, text="Meu").grid(row=1, column=4)
        ttk.Entry(general_frame, textvariable=self.meu, width=10).grid(row=1, column=5)
        ttk.Label(general_frame, text="Sigma").grid(row=1, column=6)
        ttk.Entry(general_frame, textvariable=self.sigma, width=10).grid(row=1, column=7)

        ttk.Button(general_frame, text="Add Sin", command=lambda: self.add_function('sin', waveform_frame)).grid(row=2, column=0)
        ttk.Button(general_frame, text="Add Cos", command=lambda: self.add_function('cos', waveform_frame)).grid(row=2, column=1)
        ttk.Button(general_frame, text="Plot", command=self.plot_waveform).grid(row=2, column=2)

        ttk.Button(export_frame, text="Save Config", command=self.save_config).pack()
        ttk.Button(export_frame, text="Load Config", command=self.load_config).pack()

    def add_function(self, func_type, parent):
        row = len(self.entries)
        type_var = tk.StringVar(value=func_type)
        self.type_vars.append(type_var)

        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky='w')
        self.frames.append(frame)

        amp = tk.DoubleVar(value=1)
        freq = tk.DoubleVar(value=1)
        phase = tk.DoubleVar(value=0)
        freq_change = tk.BooleanVar(value=False)
        t_naught = tk.DoubleVar(value=1.0)

        ttk.Label(frame, text=func_type).grid(row=0, column=0)
        ttk.Entry(frame, textvariable=amp, width=10).grid(row=0, column=1)
        ttk.Entry(frame, textvariable=freq, width=10).grid(row=0, column=2)
        ttk.Entry(frame, textvariable=phase, width=10).grid(row=0, column=3)

        check = ttk.Checkbutton(frame, text="Frequency Change", variable=freq_change,
                                command=lambda v=freq_change, f=frame, tn=t_naught: self.toggle_t_naught(v, f, tn))
        check.grid(row=0, column=4)

        ttk.Button(frame, text="Remove", command=lambda: self.remove_function(row)).grid(row=0, column=5)

        self.entries.append((amp, freq, phase, freq_change, t_naught))
        self.freq_change_vars.append(freq_change)
        self.t_naught_vars.append(t_naught)

    def remove_function(self, index):
        self.frames[index].destroy()
        self.entries.pop(index)
        self.type_vars.pop(index)
        self.freq_change_vars.pop(index)
        self.t_naught_vars.pop(index)
        self.frames.pop(index)

    def toggle_t_naught(self, var, frame, t_naught):
        if var.get():
            ttk.Label(frame, text="t_naught").grid(row=1, column=4)
            ttk.Entry(frame, textvariable=t_naught, width=10).grid(row=1, column=5)

    def setup_plot(self):
        self.figure, self.axes = plt.subplots(4, 1, figsize=(8, 10))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def plot_waveform(self):
        steps = int(self.step_var.get())
        dt = self.dt_var.get()
        t = np.arange(0, steps * dt, dt)
        total_time = steps * dt
        sampling_rate = 1 / dt
        self.total_time_label.config(text=f"Total time: {total_time:.4f} s")
        self.sampling_rate_label.config(text=f"Sampling rate: {sampling_rate:.2f} Hz")

        f = np.zeros_like(t)

        for i, (amp, freq, phase, freq_change, t_naught) in enumerate(self.entries):
            A = amp.get()
            w = freq.get()
            phi = phase.get()
            func_type = self.type_vars[i].get()

            if freq_change.get():
                tn = t_naught.get()
                mask = t < tn
                wn_sq = np.zeros_like(t)
                wn_sq[mask] = w / (tn ** 2 - t[mask])
                current_t = t[mask]
            else:
                wn_sq = np.full_like(t, w)
                current_t = t

            if func_type == 'sin':
                f[:len(current_t)] += A * np.sin(2 * np.pi * wn_sq[:len(current_t)] * current_t + phi)
            elif func_type == 'cos':
                f[:len(current_t)] += A * np.cos(2 * np.pi * wn_sq[:len(current_t)] * current_t + phi)

        factor = self.multiplier_factor.get()
        if self.multiplier_type.get() == "linear":
            f *= factor * t
        elif self.multiplier_type.get() == "quadratic":
            f *= factor * t ** 2
        elif self.multiplier_type.get() == "exponential":
            f *= np.exp(factor * t)
        elif self.multiplier_type.get() == "envelope":
            f *= np.exp(-((t - self.meu.get()) ** 2) / (2 * self.sigma.get() ** 2))

        for ax in self.axes:
            ax.clear()

        self.axes[0].plot(t, f)
        self.axes[0].set_title("Voltage vs Time")
        self.axes[0].set_xlabel("Time (s)")
        self.axes[0].set_ylabel("Voltage")

        real = np.fft.rfft(f, norm='forward')
        real_freq = np.fft.rfftfreq(n=steps, d=dt)
        self.axes[1].plot(real_freq, np.abs(real))
        self.axes[1].set_title("Real Fourier Transform")
        self.axes[1].set_xlabel("Frequency (Hz)")
        self.axes[1].set_ylabel("Amplitude")

        complex_fft = np.fft.fft(f, norm='forward')
        freq = np.fft.fftfreq(n=len(complex_fft), d=dt)
        self.axes[2].plot(freq, np.abs(complex_fft))
        self.axes[2].set_title("Complex Fourier Transform")
        self.axes[2].set_xlabel("Frequency (Hz)")
        self.axes[2].set_ylabel("Amplitude")

        x, y, z = signal.spectrogram(f, fs=1/dt)
        self.axes[3].pcolormesh(y, x, z, shading='gouraud')
        self.axes[3].set_title("Spectrogram")
        self.axes[3].set_xlabel("Time (s)")
        self.axes[3].set_ylabel("Frequency (Hz)")

        self.canvas.draw()

    def save_config(self):
        config = {
            'steps': self.step_var.get(),
            'dt': self.dt_var.get(),
            'multiplier_type': self.multiplier_type.get(),
            'multiplier_factor': self.multiplier_factor.get(),
            'meu': self.meu.get(),
            'sigma': self.sigma.get(),
            'entries': [
                {
                    'type': self.type_vars[i].get(),
                    'amp': e[0].get(),
                    'freq': e[1].get(),
                    'phase': e[2].get(),
                    'freq_change': e[3].get(),
                    't_naught': e[4].get()
                }
                for i, e in enumerate(self.entries)
            ]
        }
        with open("waveform_config.json", "w") as f:
            json.dump(config, f)

    def load_config(self):
        with open("waveform_config.json", "r") as f:
            config = json.load(f)
        self.step_var.set(config['steps'])
        self.dt_var.set(config['dt'])
        self.multiplier_type.set(config['multiplier_type'])
        self.multiplier_factor.set(config['multiplier_factor'])
        self.meu.set(config['meu'])
        self.sigma.set(config['sigma'])

        for frame in self.frames:
            frame.destroy()

        self.entries.clear()
        self.type_vars.clear()
        self.freq_change_vars.clear()
        self.t_naught_vars.clear()
        self.frames.clear()

        for entry in config['entries']:
            self.add_function(entry['type'], self.root.children['!notebook'].children['!frame2'])
            i = len(self.entries) - 1
            self.entries[i][0].set(entry['amp'])
            self.entries[i][1].set(entry['freq'])
            self.entries[i][2].set(entry['phase'])
            self.entries[i][3].set(entry['freq_change'])
            self.entries[i][4].set(entry['t_naught'])

if __name__ == "__main__":
    root = tk.Tk()
    app = WaveformGUI(root)
    root.mainloop()
