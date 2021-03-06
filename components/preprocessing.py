import nussl
import numpy as np
import streamlit as st
import pyroomacoustics as pra

from scipy import signal


def butterworth_filter(y: np.ndarray,
                       sr: int,
                       N: int,
                       cutoff=500.,
                       btype="lowpass"):
    b, a = signal.butter(N, cutoff / (sr / 2.), btype=btype)
    y_filtered = signal.filtfilt(b, a, y)
    return y_filtered


def preprocess_on_wave(y: np.ndarray, sr: int, audio_path: str):
    st.sidebar.markdown("#### Preprocess option")
    option = st.sidebar.selectbox(
        "process", options=["-", "normalize", "lowpass", "highpass", "bandpass", "denoise", "nussl"])
    if option == "lowpass":
        param_N = st.sidebar.number_input(
            "N", min_value=1, max_value=10, value=4, step=1)
        param_cutoff = st.sidebar.number_input(
            "cutoff", min_value=20.0, max_value=4000.0, value=500.0, step=10.0)
        filtered = butterworth_filter(
            y, sr=sr, N=param_N, cutoff=param_cutoff, btype="lowpass")
        return np.asfortranarray(filtered)
    elif option == "highpass":
        param_N = st.sidebar.number_input(
            "N", min_value=1, max_value=10, value=4, step=1)
        param_cutoff = st.sidebar.number_input(
            "cutoff",
            min_value=500.0,
            max_value=16000.0,
            value=1000.0,
            step=10.0)
        filtered = butterworth_filter(
            y, sr=sr, N=param_N, cutoff=param_cutoff, btype="highpass")
        return np.asfortranarray(filtered)
    elif option == "bandpass":
        param_N = st.sidebar.number_input(
            "N", min_value=1, max_value=10, value=4, step=1)
        upper_limit = st.sidebar.number_input(
            "upper_limit", min_value=0.0, max_value=16000.0, value=16000.0, step=10.0)
        lower_limit = st.sidebar.number_input(
            "lower_limit", min_value=0.0, max_value=16000.0, value=20.0, step=10.0)
        lowpassed = butterworth_filter(
            y, sr=sr, N=param_N, cutoff=upper_limit, btype="lowpass")
        bandpassed = butterworth_filter(
            lowpassed, sr=sr, N=param_N, cutoff=lower_limit, btype="highpass")
        return np.asfortranarray(bandpassed)
    elif option == "normalize":
        max_vol = np.abs(y).max()
        y_vol = y * 1 / (max_vol)
        return np.asfortranarray(y_vol)
    elif option == "denoise":
        frame_len = st.sidebar.number_input(
            "frame_len", min_value=1, max_value=8192, value=512, step=32)
        lpc_order = st.sidebar.number_input(
            "lpc_order", min_value=1, max_value=100, value=20, step=1)
        iterations = st.sidebar.number_input(
            "iterations", min_value=1, max_value=100, value=2, step=1)
        alpha = st.sidebar.number_input(
            "alpha", min_value=0.1, max_value=10.0, value=0.8, step=0.1)
        thresh = st.sidebar.number_input(
            "thresh", min_value=0.01, value=0.01, step=0.01, max_value=10.0)
        denoised = pra.denoise.apply_iterative_wiener(
            y, frame_len, lpc_order, iterations, alpha, thresh)
        return denoised
    elif option == "nussl":
        history = nussl.AudioSignal(audio_path)
        method = st.sidebar.selectbox(
            "Denoise method",
            options=[
                "Repet", "ICA", "FT2D", "REPETSIM", "TimbreClustering", "HPSS",
                "DUET", "PROJET"
            ])
        if method == "Repet":
            separator = nussl.separation.primitive.Repet(history)
        elif method == "ICA":
            separator = nussl.separation.factorization.ICA(history)
        elif method == "FT2D":
            separator = nussl.separation.primitive.FT2D(history)
        elif method == "REMETSIM":
            separator = nussl.separation.primitive.REMETSIM(history)
        elif method == "TimbreClustering":
            separator = nussl.separation.primitive.TimbreClustering(history, num_sources=2, n_components=50)
        elif method == "HPSS":
            separator = nussl.separation.primitive.HPSS(history)
        elif method == "DUET":
            separator = nussl.separation.spatial.Duet(history, num_sources=2)
        elif method == "PROJET":
            separator = nussl.separation.spatial.Projet(history, num_sources=2)

        estimates = separator()
        foreground = estimates[1].audio_data[0]
        return foreground
    else:
        return None
