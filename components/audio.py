import audioread
import warnings

import librosa
import pandas as pd
import streamlit as st

from pathlib import Path, PosixPath
from typing import List


@st.cache
def read_audio(path: str):
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
        if not base_path.exists():
            st.warning("Specified directory does not exists")
        else:
            st.warning(
                "Specified directory does not contain readable audio file.")
            glob_result = list(base_path.glob("*"))
            st.write("Directories in this directory")
            dirs = [path.name for path in glob_result if path.is_dir()]
            st.write(dirs)
        return None


def open_specified_audio(audio_files: List[PosixPath]):
    audio_file_paths = [str(path) for path in audio_files]
    path = st.sidebar.selectbox("File to read", options=audio_file_paths)

    with open(path, "rb") as f:
        audio_bytes = f.read()

    st.audio(audio_bytes, format="audio/mp3")


def check_audio_option(df: pd.DataFrame):
    st.sidebar.subheader("Check audio settings")

    audio_files = configure_audio_dir(df)
    if audio_files is not None:
        open_specified_audio(audio_files)
