import librosa.display as display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def waveplot(y: np.ndarray, sr: int):
    plot_wave = st.checkbox("Waveplot")
    if plot_wave:
        plt.grid(True)
        display.waveplot(y, sr=sr)

        st.pyplot()
