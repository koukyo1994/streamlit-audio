import librosa
import librosa.display as display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def waveplot(y: np.ndarray, sr: int):
    plot_wave = st.checkbox("Waveplot")
    if plot_wave:
        st.sidebar.markdown("#### Waveplot settings")
        start_second = st.sidebar.number_input(
            "start second",
            min_value=0,
            max_value=len(y) // sr,
            value=0,
            step=1,
            key="waveplot_start")
        end_second = st.sidebar.number_input(
            "end second",
            min_value=0,
            max_value=len(y) // sr,
            value=len(y) // sr,
            step=1,
            key="waveplot_end")

        start_index = start_second * sr
        if end_second == len(y) // sr:
            end_index = len(y)
        else:
            end_index = end_second * sr
        plt.figure(figsize=(12, 4))
        plt.grid(True)
        display.waveplot(y[start_index:end_index], sr=sr)

        st.pyplot()


@st.cache
def melspectrogram(y: np.ndarray, params: dict, log=True):
    melspec = librosa.feature.melspectrogram(y=y, **params)
    if log:
        melspec = librosa.power_to_db(melspec)
    return melspec


def specshow(y: np.ndarray, sr: int):
    plot_spectrogram = st.checkbox("Spectrogram plot")
    if plot_spectrogram:
        st.sidebar.markdown("#### Spectrogram plot settings")
        start_second = st.sidebar.number_input(
            "start second",
            min_value=0,
            max_value=len(y) // sr,
            value=0,
            step=1,
            key="specshow_start")
        end_second = st.sidebar.number_input(
            "end second",
            min_value=0,
            max_value=len(y) // sr,
            value=len(y) // sr,
            step=1,
            key="specshow_end")
        start_index = start_second * sr
        if end_second == len(y) // sr:
            end_index = len(y)
        else:
            end_index = end_second * sr
        y_plot = y[start_index:end_index]

        st.sidebar.markdown("##### Melspectrogram parameters")
        n_fft = st.sidebar.number_input(
            "n_fft", min_value=64, max_value=8192, value=1024, step=64)
        hop_length = st.sidebar.number_input(
            "hop_length", min_value=1, max_value=2048, value=320, step=10)
        n_mels = st.sidebar.number_input(
            "n_mels", min_value=1, max_value=512, value=64, step=16)
        fmin = st.sidebar.number_input(
            "fmin", min_value=1, max_value=8192, value=20, step=100)
        fmax = st.sidebar.number_input(
            "fmax", min_value=4000, max_value=44100, value=14000, step=100)
        log = st.sidebar.checkbox("apply log")

        melspec_params = {
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": n_mels,
            "fmin": fmin,
            "fmax": fmax,
            "sr": sr
        }

        if st.button("Show melspectrogram"):
            with st.spinner("Calculating melspectrogram"):
                melspec = melspectrogram(y_plot, melspec_params, log)

            with st.spinner("Plotting"):
                plt.figure(figsize=(12, 4))
                display.specshow(
                    melspec,
                    sr=sr,
                    hop_length=hop_length,
                    x_axis="time",
                    y_axis="mel",
                    fmin=fmin,
                    fmax=fmax)

            st.pyplot()
