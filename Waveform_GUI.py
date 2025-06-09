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

        style = ttk.Style(self.root)
        style.theme_use("xpnative")

        self.entries = []
        self.type_vars = []
        self.freq_change_vars = []
        self.t_naught_vars = []

        self.total_time_var = tk.DoubleVar()
        self.sampling_rate_var = tk.DoubleVar()

        self.linear_factor = tk.DoubleVar(value=1.0)
        self.quadratic_factor = tk.DoubleVar(value=1.0)
        self.exponential_factor = tk.DoubleVar(value=1.0)
        self.envelope_factor = tk.DoubleVar(value=1.0)
        self.meu = tk.DoubleVar(value=0.5)
        self.sigma = tk.DoubleVar(value=0.1)



        self.setup_controls()
        self.setup_plot()

    def setup_controls(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        general_frame = ttk.Frame(notebook)
        self.waveform_frame = ttk.Frame(notebook)
        export_frame = ttk.Frame(notebook)

        notebook.add(general_frame, text='General Settings')
        notebook.add(self.waveform_frame, text='Waveform Components')
        notebook.add(export_frame, text='Export Options')

        # Add buttons in waveform tab
        # Add top control row
        control_frame = ttk.Frame(self.waveform_frame)
        control_frame.grid(row=0, column=0, columnspan=6, sticky='w')

        ttk.Button(control_frame, text="Add Sin", command=lambda: self.add_function('sin')).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(control_frame, text="Add Cos", command=lambda: self.add_function('cos')).grid(row=0, column=1, padx=5, pady=5)

        # Add header labels
        ttk.Label(self.waveform_frame, text="Type").grid(row=1, column=0, padx=5)
        ttk.Label(self.waveform_frame, text="Amplitude").grid(row=1, column=1, padx=5)
        ttk.Label(self.waveform_frame, text="Frequency").grid(row=1, column=2, padx=5)
        ttk.Label(self.waveform_frame, text="Phase").grid(row=1, column=3, padx=5)
        ttk.Label(self.waveform_frame, text="Options").grid(row=1, column=4, padx=5)
        ttk.Label(self.waveform_frame, text="").grid(row=1, column=5)  # Spacer for Remove button

        self.current_row = 2  # Start dynamic rows below the header


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
        ttk.Label(general_frame, text="Total time (s)").grid(row=0, column=4)
        ttk.Entry(general_frame, textvariable=self.total_time_var, width=10).grid(row=0, column=5)
        ttk.Label(general_frame, text="Sampling rate (Hz)").grid(row=0, column=6)
        ttk.Entry(general_frame, textvariable=self.sampling_rate_var, width=10).grid(row=0, column=7)

        # Multiplier type selector
        ttk.Label(general_frame, text="Multiplier Type").grid(row=1, column=0)
        self.multiplier_type = tk.StringVar(value="none")
        type_menu = ttk.Combobox(general_frame, textvariable=self.multiplier_type,
                                values=["none", "linear", "quadratic", "exponential", "envelope"])
        type_menu.grid(row=1, column=1)
        type_menu.bind("<<ComboboxSelected>>", lambda e: self.update_multiplier_inputs())

        # Frame to hold dynamic multiplier options
        self.multiplier_frame = ttk.Frame(general_frame)
        self.multiplier_frame.grid(row=1, column=2, columnspan=6, sticky='w')

        ttk.Button(general_frame, text="Plot", command=self.plot_waveform).grid(row=2, column=2)

        ttk.Button(export_frame, text="Save Config", command=self.save_config).pack()
        ttk.Button(export_frame, text="Load Config", command=self.load_config).pack()

        self.dt_var.trace_add("write", lambda *args: self.update_from_dt())
        self.step_var.trace_add("write", lambda *args: self.update_from_steps())
        self.total_time_var.trace_add("write", lambda *args: self.update_from_total_time())
        self.sampling_rate_var.trace_add("write", lambda *args: self.update_from_sampling_rate())

        self.update_multiplier_inputs()


    def add_function(self, func_type):
        row = self.current_row
        self.current_row += 1

        amp = tk.DoubleVar(value=1)
        freq = tk.DoubleVar(value=1)
        phase = tk.DoubleVar(value=0)
        freq_change = tk.BooleanVar(value=False)
        t_naught = tk.DoubleVar(value=1.0)
        type_var = tk.StringVar(value=func_type)

        self.entries.append((amp, freq, phase, freq_change, t_naught))
        self.type_vars.append(type_var)
        self.freq_change_vars.append(freq_change)
        self.t_naught_vars.append(t_naught)

        ttk.Label(self.waveform_frame, text=func_type).grid(row=row, column=0)
        ttk.Entry(self.waveform_frame, textvariable=amp, width=10).grid(row=row, column=1)
        ttk.Entry(self.waveform_frame, textvariable=freq, width=10).grid(row=row, column=2)
        ttk.Entry(self.waveform_frame, textvariable=phase, width=10).grid(row=row, column=3)

        def remove_this():
            for widget in self.waveform_frame.grid_slaves(row=row):
                widget.destroy()
            self.entries[row - 2] = None  # Leave hole in list
            self.type_vars[row - 2] = None
            self.freq_change_vars[row - 2] = None
            self.t_naught_vars[row - 2] = None

        ttk.Checkbutton(self.waveform_frame, text="Freq Change", variable=freq_change,
                        command=lambda: self.toggle_t_naught(freq_change, row, t_naught)).grid(row=row, column=4)

        ttk.Button(self.waveform_frame, text="Remove", command=remove_this).grid(row=row, column=5)


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

    def update_from_dt(self):
        try:
            dt = self.dt_var.get()
            steps = int(self.total_time_var.get() / dt)
            sampling_rate = 1 / dt

            self.step_var.set(steps)
            self.sampling_rate_var.set(sampling_rate)
        except:
            pass

    def update_from_steps(self):
        try:
            dt = self.dt_var.get()
            steps = int(self.step_var.get())
            total_time = steps * dt
            sampling_rate = 1 / dt

            self.total_time_var.set(total_time)
            self.sampling_rate_var.set(sampling_rate)
        except:
            pass

    def update_from_total_time(self):
        try:
            dt = self.dt_var.get()
            total_time = self.total_time_var.get()
            steps = int(total_time / dt)

            self.step_var.set(steps)
        except:
            pass

    def update_from_sampling_rate(self):
        try:
            rate = self.sampling_rate_var.get()
            dt = 1 / rate
            self.dt_var.set(dt)
        except:
            pass


    def plot_waveform(self):
        # force re-sync
        self.update_from_steps()

        steps = int(self.step_var.get())
        dt = self.dt_var.get()

        t = np.arange(0, steps * dt, dt)
        total_time = steps * dt
        sampling_rate = 1 / dt
    

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

        mode = self.multiplier_type.get()

        if mode == "linear":
            f *= self.linear_factor.get() * t

        elif mode == "quadratic":
            f *= self.linear_factor.get() * t + self.quadratic_factor.get() * t ** 2

        elif mode == "exponential":
            f *= np.exp(self.exponential_factor.get() * t)

        elif mode == "envelope":
            mu = self.meu.get()
            sigma = self.sigma.get()
            A = self.envelope_factor.get()
            f *= A * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))


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
        
                # Clear all waveform UI elements
        for widget in self.waveform_frame.winfo_children():
            widget.destroy()

        self.entries.clear()
        self.type_vars.clear()
        self.freq_change_vars.clear()
        self.t_naught_vars.clear()

        # Rebuild header
        ttk.Button(self.waveform_frame, text="Add Sin", command=lambda: self.add_function('sin')).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        ttk.Button(self.waveform_frame, text="Add Cos", command=lambda: self.add_function('cos')).grid(row=0, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(self.waveform_frame, text="Amplitude").grid(row=1, column=1, padx=5)
        ttk.Label(self.waveform_frame, text="Frequency").grid(row=1, column=2, padx=5)
        ttk.Label(self.waveform_frame, text="Phase").grid(row=1, column=3, padx=5)


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

    def update_multiplier_inputs(self):
    # Clear old widgets
        for widget in self.multiplier_frame.winfo_children():
            widget.destroy()

        mode = self.multiplier_type.get()

        if mode == "linear":
            ttk.Label(self.multiplier_frame, text="Linear Factor").grid(row=0, column=0)
            ttk.Entry(self.multiplier_frame, textvariable=self.linear_factor, width=10).grid(row=0, column=1)

        elif mode == "quadratic":
            ttk.Label(self.multiplier_frame, text="Linear Factor").grid(row=0, column=0)
            ttk.Entry(self.multiplier_frame, textvariable=self.linear_factor, width=10).grid(row=0, column=1)
            ttk.Label(self.multiplier_frame, text="Quadratic Factor").grid(row=0, column=2)
            ttk.Entry(self.multiplier_frame, textvariable=self.quadratic_factor, width=10).grid(row=0, column=3)

        elif mode == "exponential":
            ttk.Label(self.multiplier_frame, text="Exponential Factor").grid(row=0, column=0)
            ttk.Entry(self.multiplier_frame, textvariable=self.exponential_factor, width=10).grid(row=0, column=1)

        elif mode == "envelope":
            ttk.Label(self.multiplier_frame, text="Factor (A)").grid(row=0, column=0)
            ttk.Entry(self.multiplier_frame, textvariable=self.envelope_factor, width=10).grid(row=0, column=1)
            ttk.Label(self.multiplier_frame, text="Mu").grid(row=0, column=2)
            ttk.Entry(self.multiplier_frame, textvariable=self.meu, width=10).grid(row=0, column=3)
            ttk.Label(self.multiplier_frame, text="Sigma").grid(row=0, column=4)
            ttk.Entry(self.multiplier_frame, textvariable=self.sigma, width=10).grid(row=0, column=5)

    def toggle_t_naught(self, var, row, t_naught):
        for widget in self.waveform_frame.grid_slaves(row=row+1):
            widget.destroy()

        if var.get():
            ttk.Label(self.waveform_frame, text="t_naught").grid(row=row+1, column=4)
            ttk.Entry(self.waveform_frame, textvariable=t_naught, width=10).grid(row=row+1, column=5)



if __name__ == "__main__":
    root = tk.Tk()
    app = WaveformGUI(root)
    root.mainloop() 
    