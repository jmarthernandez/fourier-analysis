import numpy as np
from scipy import signal

from dash import Dash, dcc, html, Input, Output, State, ctx, MATCH, ALL
import plotly.graph_objs as go

app = Dash(__name__)
server = app.server

# App layout
app.layout = html.Div([
    html.H1("Waveform Synthesizer in Dash"),
    html.Div([
        html.Label("Amplitude"),
        dcc.Input(
            id="amplitude",
            type="number",
            min=0.1,
            max=5,
            step=0.1,
            value=1,
            style={"width": "100px"}  # adjust width as you like
        ),

        html.Br(), html.Br(),

        html.Label("Frequency (Hz)"),
        dcc.Input(
            id="frequency",
            type="number",
            min=0.5,
            max=10,
            step=0.5,
            value=2,
            style={"width": "100px"}
        ),

        html.Br(), html.Br(),

        html.Label("Phase (rad)"),
        dcc.Input(
            id="phase",
            type="number",
            min=0,
            max=2 * np.pi,
            step=0.1,
            value=0,
            style={"width": "100px"}
        ),
        html.Br(), html.Br(),
        html.Label("Waveform Type"),
        dcc.Dropdown(
            id="wave-type",
            options=[
                {"label": "Sine",     "value": "sine"},
                {"label": "Square",   "value": "square"},
                {"label": "Triangle", "value": "triangle"},
                {"label": "Sawtooth", "value": "sawtooth"},
            ],
            value="sine",
            clearable=False,
            style={"width": "100px"}
        ),
        html.Br(),
        html.Button("Add Waveform", id="add-btn", n_clicks=0),
        html.H4("Added Waveforms:"),
        html.Div(id="saved-waveforms"),
        dcc.Store(id="waveform-store", data=[]),
    ], style={"width": "40%", "display": "inline-block", "verticalAlign": "top"}),

    html.Div([
        dcc.Graph(id="waveform-graph"),
        dcc.Graph(id="spectrum-graph"),
        dcc.Graph(id="autocorr-graph"),
        dcc.Graph(id="spectrogram-graph"),
    ], style={"width": "55%", "display": "inline-block", "paddingLeft": "20px"})
])


# 1) Add a new waveform
@app.callback(
    Output("waveform-store", "data"),
    Output("saved-waveforms", "children"),
    Input("add-btn", "n_clicks"),
    State("amplitude", "value"),
    State("frequency", "value"),
    State("phase", "value"),
    State("wave-type", "value"),
    State("waveform-store", "data"),
    prevent_initial_call=True
)
def add_wave(n_clicks, amp, freq, phase, wtype, store):
    # Copy & append
    store = store.copy()
    store.append({"amp": amp, "freq": freq, "phase": phase, "type": wtype})

    # Build list of children with Delete buttons
    children = []
    for i, w in enumerate(store):
        desc = f"{w['type'].title()} | Amp={w['amp']}, Freq={w['freq']}, Phase={w['phase']:.2f}"
        children.append(
            html.Div([
                html.Span(desc, style={"marginRight": "10px"}),
                html.Button(
                    "Delete",
                    id={"type": "delete-btn", "index": i},
                    n_clicks=0,
                    style={"fontSize": "0.8em"}
                )
            ], style={"padding": "4px 0"})
        )

    return store, children


# 2) Delete a waveform via pattern-matching callback
@app.callback(
    Output("waveform-store", "data"),
    Output("saved-waveforms", "children"),
    Input({"type": "delete-btn", "index": ALL}, "n_clicks"),
    State("waveform-store", "data"),
    prevent_initial_call=True
)
def delete_wave(n_clicks_list, store):
    triggered = ctx.triggered_id
    if not triggered:
        raise dash.exceptions.PreventUpdate

    idx = triggered["index"]
    # Remove the selected waveform
    updated = [w for i, w in enumerate(store) if i != idx]

    # Rebuild children
    children = []
    for i, w in enumerate(updated):
        desc = f"{w['type'].title()} | Amp={w['amp']}, Freq={w['freq']}, Phase={w['phase']:.2f}"
        children.append(
            html.Div([
                html.Span(desc, style={"marginRight": "10px"}),
                html.Button(
                    "Delete",
                    id={"type": "delete-btn", "index": i},
                    n_clicks=0,
                    style={"fontSize": "0.8em"}
                )
            ], style={"padding": "4px 0"})
        )

    return updated, children


# 3) Plot everything based on stored waveforms
@app.callback(
    Output("waveform-graph", "figure"),
    Output("spectrum-graph", "figure"),
    Output("autocorr-graph", "figure"),
    Output("spectrogram-graph", "figure"),
    Input("waveform-store", "data")
)
def update_all(wave_list):
    dt = 0.001
    t = np.arange(0, 1, dt)
    y = np.zeros_like(t)

    # Sum each stored waveform
    for wave in wave_list:
        amp, freq, phase, wtype = wave.values()
        if wtype == "sine":
            y += amp * np.sin(2*np.pi*freq*t + phase)
        elif wtype == "square":
            y += amp * signal.square(2*np.pi*freq*t + phase)
        elif wtype == "triangle":
            y += amp * signal.sawtooth(2*np.pi*freq*t + phase, width=0.5)
        else:
            y += amp * signal.sawtooth(2*np.pi*freq*t + phase)

    # Time-domain plot
    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(x=t, y=y, mode="lines"))
    fig_wave.update_layout(
        title="Combined Waveform",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude"
    )

    # Power spectrum
    Y = np.fft.fft(y)
    freq_axis = np.fft.fftfreq(len(t), d=dt)
    mask = freq_axis >= 0
    fig_spec = go.Figure()
    fig_spec.add_trace(go.Bar(x=freq_axis[mask], y=np.abs(Y[mask])))
    fig_spec.update_layout(
        title="Power Spectrum",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude"
    )

    # Autocorrelation
    corr = np.correlate(y, y, mode="full")
    lags = np.arange(-len(y) + 1, len(y))
    fig_auto = go.Figure()
    fig_auto.add_trace(go.Scatter(x=lags * dt, y=corr, mode="lines"))
    fig_auto.update_layout(
        title="Autocorrelation",
        xaxis_title="Lag (s)",
        yaxis_title="Correlation"
    )

    # Spectrogram
    f_s, t_s, Sxx = signal.spectrogram(y, fs=1/dt)
    fig_specg = go.Figure(data=go.Heatmap(
        x=t_s, y=f_s, z=10 * np.log10(Sxx),
        colorscale="Viridis"
    ))
    fig_specg.update_layout(
        title="Spectrogram (dB)",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)"
    )

    return fig_wave, fig_spec, fig_auto, fig_specg


if __name__ == "__main__":
    app.run(debug=True)
