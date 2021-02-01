import librosa
import librosa.display as display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import streamlit as st


def waveplot(y: np.ndarray, sr: int, processed=None, tp: pd.DataFrame=None, fp: pd.DataFrame=None):
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
        fig = plt.figure(figsize=(12, 4))
        plt.grid(True)
        display.waveplot(y[start_index:end_index], sr=sr, alpha=0.5)
        if processed is not None:
            display.waveplot(
                processed[start_index:end_index],
                sr=sr,
                alpha=0.5,
                color="red")
        if tp is not None and len(tp) > 0:
            for _, row in tp.iterrows():
                plt.axvspan(row["t_min"], row["t_max"], color="g", alpha=0.5, label=str(row["species_id"]))

        if fp is not None and len(fp) > 0:
            for _, row in fp.iterrows():
                plt.axvspan(row["t_min"], row["t_max"], color="r", alpha=0.5, label=str(row["species_id"]))
        plt.legend()

        st.pyplot(fig)


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
            end_second = len(y) / sr
        else:
            end_index = end_second * sr
        events_in_period = events.query(
            f"onset >= {start_second} & offset <= {end_second}")
        uniq_labels = events_in_period["ebird_code"].unique().tolist()
        fig = plt.figure(figsize=(12, 4))
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

        st.pyplot(fig)


@st.cache
def melspectrogram(y: np.ndarray, params: dict, log=True):
    melspec = librosa.feature.melspectrogram(y=y, **params)
    if log:
        melspec = librosa.power_to_db(melspec)
    return melspec


@st.cache
def spectrogram(y: np.ndarray, params: dict, log=True):
    spec = librosa.stft(y, **params)
    if log:
        spec = librosa.power_to_db(spec)
    return spec


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
            f"onset >= {start_second} & offset <= {end_second}")
        uniq_labels = events_in_period["ebird_code"].unique().tolist()

        st.sidebar.markdown("##### (Mel)spectrogram parameters")
        mel = st.sidebar.checkbox("Mel scale", value=True)

        n_fft = st.sidebar.number_input(
            "n_fft", min_value=64, max_value=8192, value=1024, step=64)
        hop_length = st.sidebar.number_input(
            "hop_length", min_value=1, max_value=2048, value=320, step=10)
        if mel:
            n_mels = st.sidebar.number_input(
                "n_mels", min_value=1, max_value=512, value=64, step=16)
            fmin = st.sidebar.number_input(
                "fmin", min_value=1, max_value=8192, value=20, step=100)
            fmax = st.sidebar.number_input(
                "fmax", min_value=4000, max_value=44100, value=14000, step=100)
        log = st.sidebar.checkbox("apply log", value=True)

        if mel:
            melspec_params = {
                "n_fft": n_fft,
                "hop_length": hop_length,
                "n_mels": n_mels,
                "fmin": fmin,
                "fmax": fmax,
                "sr": sr
            }
        else:
            spec_params = {
                "n_fft": n_fft,
                "hop_length": hop_length
            }

        if st.button("Show melspectrogram"):
            with st.spinner("Calculating melspectrogram"):
                if mel:
                    spec = melspectrogram(y_plot, melspec_params, log)
                else:
                    spec = spectrogram(y_plot, spec_params, log)
                if y_processed is not None:
                    if mel:
                        spec_processed = melspectrogram(y_plot_processed,
                                                        melspec_params, log)
                    else:
                        spec_processed = spectrogram(y_plot_processed,
                                                     spec_params, log)

            height, width = spec.shape
            st.write(f"{height} x {width} matrix")
            if y_processed is not None:
                with st.spinner("Plotting"):
                    fig = plt.figure(figsize=(12, 8))
                    ax1 = fig.add_subplot(2, 1, 1)
                    if mel:
                        display.specshow(
                            spec,
                            sr=sr,
                            hop_length=hop_length,
                            x_axis="time",
                            y_axis="mel",
                            fmin=fmin,
                            fmax=fmax,
                            ax=ax1)
                    else:
                        display.specshow(
                            spec,
                            sr=sr,
                            hop_length=hop_length,
                            x_axis="time",
                            y_axis="linear",
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
                    if mel:
                        display.specshow(
                            spec_processed,
                            sr=sr,
                            hop_length=hop_length,
                            x_axis="time",
                            y_axis="mel",
                            fmin=fmin,
                            fmax=fmax,
                            ax=ax2)
                    else:
                        display.specshow(
                            spec_processed,
                            sr=sr,
                            hop_length=hop_length,
                            x_axis="time",
                            y_axis="linear",
                            ax=ax2)

            else:
                with st.spinner("Plotting"):
                    fig = plt.figure(figsize=(12, 4))
                    if mel:
                        display.specshow(
                            spec,
                            sr=sr,
                            hop_length=hop_length,
                            x_axis="time",
                            y_axis="mel",
                            fmin=fmin,
                            fmax=fmax)
                        plt.colorbar()
                    else:
                        display.specshow(
                            spec,
                            sr=sr,
                            hop_length=hop_length,
                            x_axis="time",
                            y_axis="linear")
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

            st.pyplot(fig)


def specshow(y: np.ndarray, sr: int, y_processed=None, tp: pd.DataFrame=None, fp: pd.DataFrame=None):
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

        st.sidebar.markdown("##### (Mel)spectrogram parameters")
        mel = st.sidebar.checkbox("Mel scale", value=True)

        n_fft = st.sidebar.number_input(
            "n_fft", min_value=64, max_value=8192, value=1024, step=64)
        hop_length = st.sidebar.number_input(
            "hop_length", min_value=1, max_value=2048, value=320, step=10)
        if mel:
            n_mels = st.sidebar.number_input(
                "n_mels", min_value=1, max_value=512, value=64, step=16)
            fmin = st.sidebar.number_input(
                "fmin", min_value=1, max_value=8192, value=20, step=100)
            fmax = st.sidebar.number_input(
                "fmax", min_value=4000, max_value=44100, value=14000, step=100)
        log = st.sidebar.checkbox("apply log", value=True)

        if mel:
            melspec_params = {
                "n_fft": n_fft,
                "hop_length": hop_length,
                "n_mels": n_mels,
                "fmin": fmin,
                "fmax": fmax,
                "sr": sr
            }
        else:
            spec_params = {
                "n_fft": n_fft,
                "hop_length": hop_length
            }

        if st.button("Show melspectrogram"):
            with st.spinner("Calculating melspectrogram"):
                if mel:
                    spec = melspectrogram(y_plot, melspec_params, log)
                else:
                    spec = spectrogram(y_plot, spec_params, log)
                if y_processed is not None:
                    if mel:
                        spec_processed = melspectrogram(y_plot_processed,
                                                        melspec_params, log)
                    else:
                        spec_processed = spectrogram(y_plot_processed,
                                                     spec_params, log)

            height, width = spec.shape
            st.write(f"{height} x {width} matrix")
            if y_processed is not None:
                with st.spinner("Plotting"):
                    fig = plt.figure(figsize=(12, 8))
                    ax1 = fig.add_subplot(2, 1, 1)
                    if mel:
                        display.specshow(
                            spec,
                            sr=sr,
                            hop_length=hop_length,
                            x_axis="time",
                            y_axis="mel",
                            fmin=fmin,
                            fmax=fmax,
                            ax=ax1)
                    else:
                        display.specshow(
                            spec,
                            sr=sr,
                            hop_length=hop_length,
                            x_axis="time",
                            y_axis="linear",
                            ax=ax1)

                    ax2 = fig.add_subplot(2, 1, 2)
                    if mel:
                        display.specshow(
                            spec_processed,
                            sr=sr,
                            hop_length=hop_length,
                            x_axis="time",
                            y_axis="mel",
                            fmin=fmin,
                            fmax=fmax,
                            ax=ax2)
                    else:
                        display.specshow(
                            spec_processed,
                            sr=sr,
                            hop_length=hop_length,
                            x_axis="time",
                            y_axis="linear",
                            ax=ax2)
            else:
                with st.spinner("Plotting"):
                    fig = plt.figure(figsize=(12, 4))
                    ax = plt.axes()
                    if mel:
                        display.specshow(
                            spec,
                            sr=sr,
                            hop_length=hop_length,
                            x_axis="time",
                            y_axis="mel",
                            fmin=fmin,
                            fmax=fmax)
                        plt.colorbar()
                    else:
                        display.specshow(
                            spec,
                            sr=sr,
                            hop_length=hop_length,
                            x_axis="time",
                            y_axis="linear")
                        plt.colorbar()
                    if tp is not None and len(tp) > 0:
                        for _, row in tp.iterrows():
                            rect = patches.Rectangle(
                                (row["t_min"], row["f_min"]),
                                row["t_max"] - row["t_min"],
                                row["f_max"] - row["f_min"],
                                linewidth=1, 
                                edgecolor="g",
                                facecolor="g",
                                alpha=0.5,
                                label="tp")
                            ax.add_patch(rect)
                    if fp is not None and len(fp) > 0:
                        for _, row in fp.iterrows():
                            rect = patches.Rectangle(
                                (row["t_min"], row["f_min"]),
                                row["t_max"] - row["t_min"],
                                row["f_max"] - row["f_min"],
                                linewidth=1,
                                edgecolor="r",
                                facecolor="r",
                                alpha=0.5,
                                label="fp")
                            ax.add_patch(rect)
            st.pyplot(fig)
