import audioread
import warnings

import librosa
import pandas as pd
import streamlit as st

from pathlib import Path


@st.cache
def read_audio(path: Path):
    with audioread.audio_open(path) as f:
        sr = f.samplerate
    warnings.filterwarnings("ignore")
    y, _ = librosa.load(path, sr=sr)
    return y, sr


def configure_audio_dir(df: pd.DataFrame):
    st.sidebar.markdown("#### File Directory Configuration")

    base_dir = st.sidebar.text_input("Base directory")
    base_path = Path(base_dir)

    has_mp3 = len(list(base_path.glob("*.mp3"))) != 0
    has_wav = len(list(base_path.glob("*.wav"))) != 0

    if has_mp3:
        return list(base_path.glob("*.mp3"))
    elif has_wav:
        return list(base_path.glob("*.wav"))
    else:
        st.warning("Specified directory does not contain readable audio file.")
        return None


def check_audio_option(df: pd.DataFrame):
    st.sidebar.subheader("Check audio settings")

    audio_files = configure_audio_dir(df)
