import librosa
import librosa.display as display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def waveplot(y: np.ndarray, sr: int, processed=None):
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
        display.waveplot(y[start_index:end_index], sr=sr, alpha=0.5)
        if processed is not None:
            display.waveplot(
                processed[start_index:end_index],
                sr=sr,
                alpha=0.5,
                color="red")

        st.pyplot()


def waveplot_with_annotation(y: np.ndarray,
                             sr: int,
                             annotation: pd.DataFrame,
                             filename: str,
                             processed=None):
    plot_wave = st.checkbox("Waveplot")
    if filename.endswith(".mp3"):
        filename = filename.replace(".mp3", ".wav")
    events = annotation.query(f"filename == '{filename}'")
    colors = [
        "#bf6565", "#ac7ceb", "#e3e176", "#f081e1", "#e8cb6b", "#25b4db",
        "#fa787e", "#a9f274", "#1d7335", "#797fb3"
    ]
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
        events_in_period = events.query(
            f"onset > {start_second} & offset < {end_second}")
        uniq_labels = events_in_period["ebird_code"].unique().tolist()
        plt.figure(figsize=(12, 4))
        plt.grid(True)
        display.waveplot(y[start_index:end_index], sr=sr, alpha=0.5)

        used_color = []  # type: ignore
        for i, event in events_in_period.iterrows():
            onset = event.onset
            offset = event.offset
            color = colors[uniq_labels.index(event.ebird_code)]
            if color not in used_color:
                label = event.ebird_code
                used_color.append(color)
            else:
                label = "_" + event.ebird_code
            plt.axvspan(onset, offset, facecolor=color, alpha=0.5, label=label)

        plt.legend()

        if processed is not None:
            display.waveplot(
                processed[start_index:end_index],
                sr=sr,
                alpha=0.5,
                color="red")

        st.pyplot()


@st.cache
def melspectrogram(y: np.ndarray, params: dict, log=True):
    melspec = librosa.feature.melspectrogram(y=y, **params)
    if log:
        melspec = librosa.power_to_db(melspec)
    return melspec


def specshow_with_annotation(y: np.ndarray,
                             sr: int,
                             annotation: pd.DataFrame,
                             filename: str,
                             y_processed=None):
    plot_spectrogram = st.checkbox("Spectrogram plot")
    if filename.endswith(".mp3"):
        filename = filename.replace(".mp3", ".wav")
    events = annotation.query(f"filename == '{filename}'")
    colors = [
        "#bf6565", "#ac7ceb", "#e3e176", "#f081e1", "#e8cb6b", "#25b4db",
        "#fa787e", "#a9f274", "#1d7335", "#797fb3"
    ]
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
        if y_processed is not None:
            y_plot_processed = y_processed[start_index:end_index]
        events_in_period = events.query(
            f"onset > {start_second} & offset < {end_second}")
        uniq_labels = events_in_period["ebird_code"].unique().tolist()

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
                if y_processed is not None:
                    melspec_processed = melspectrogram(y_plot_processed,
                                                       melspec_params, log)
            height, width = melspec.shape
            st.write(f"{height} x {width} matrix")
            if y_processed is not None:
                with st.spinner("Plotting"):
                    fig = plt.figure(figsize=(12, 8))
                    ax1 = fig.add_subplot(2, 1, 1)
                    display.specshow(
                        melspec,
                        sr=sr,
                        hop_length=hop_length,
                        x_axis="time",
                        y_axis="mel",
                        fmin=fmin,
                        fmax=fmax,
                        ax=ax1)

                    used_color = []  # type: ignore
                    for i, event in events_in_period.iterrows():
                        onset = event.onset
                        offset = event.offset
                        color = colors[uniq_labels.index(event.ebird_code)]
                        if color not in used_color:
                            label = event.ebird_code
                            used_color.append(color)
                        else:
                            label = "_" + event.ebird_code
                        ax1.axvspan(
                            onset,
                            offset,
                            facecolor=color,
                            alpha=0.5,
                            label=label)

                    ax1.legend()

                    ax2 = fig.add_subplot(2, 1, 2)
                    display.specshow(
                        melspec_processed,
                        sr=sr,
                        hop_length=hop_length,
                        x_axis="time",
                        y_axis="mel",
                        fmin=fmin,
                        fmax=fmax,
                        ax=ax2)
                    # fig.colorbar()
            else:
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
                    plt.colorbar()

                    used_color = []  # type: ignore
                    for i, event in events_in_period.iterrows():
                        onset = event.onset
                        offset = event.offset
                        color = colors[uniq_labels.index(event.ebird_code)]
                        if color not in used_color:
                            label = event.ebird_code
                            used_color.append(color)
                        else:
                            label = "_" + event.ebird_code
                        plt.axvspan(
                            onset,
                            offset,
                            facecolor=color,
                            alpha=0.5,
                            label=label)

                        plt.legend()

            st.pyplot()


def specshow(y: np.ndarray, sr: int, y_processed=None):
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
        if y_processed is not None:
            y_plot_processed = y_processed[start_index:end_index]

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
                if y_processed is not None:
                    melspec_processed = melspectrogram(y_plot_processed,
                                                       melspec_params, log)
            height, width = melspec.shape
            st.write(f"{height} x {width} matrix")
            if y_processed is not None:
                with st.spinner("Plotting"):
                    fig = plt.figure(figsize=(12, 8))
                    ax1 = fig.add_subplot(2, 1, 1)
                    display.specshow(
                        melspec,
                        sr=sr,
                        hop_length=hop_length,
                        x_axis="time",
                        y_axis="mel",
                        fmin=fmin,
                        fmax=fmax,
                        ax=ax1)

                    ax2 = fig.add_subplot(2, 1, 2)
                    display.specshow(
                        melspec_processed,
                        sr=sr,
                        hop_length=hop_length,
                        x_axis="time",
                        y_axis="mel",
                        fmin=fmin,
                        fmax=fmax,
                        ax=ax2)
                    # fig.colorbar()
            else:
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
                    plt.colorbar()

            st.pyplot()
