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


def preprocess_on_wave(y: np.ndarray, sr: int):
    st.sidebar.markdown("#### Preprocess option")
    option = st.sidebar.selectbox(
        "process", options=["-", "lowpass", "highpass", "denoise"])
    if option == "lowpass":
        param_N = st.sidebar.number_input(
            "N", min_value=1, max_value=10, value=4, step=1)
        param_cutoff = st.sidebar.number_input(
            "cutoff", min_value=20.0, max_value=4000.0, value=500.0, step=10.0)
        filtered = butterworth_filter(
            y, sr=sr, N=param_N, cutoff=param_cutoff, btype="lowpass")
        return filtered
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
        return filtered
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
    else:
        return None
