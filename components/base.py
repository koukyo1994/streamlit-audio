import streamlit as st

from pathlib import Path


def write_audio_info_to_sidebar(path: Path, info: dict):
    filename = path.name
    st.sidebar.subheader(f"Audio file: {filename}")
    st.sidebar.markdown("#### Basic info")

    for key, value in info.items():
        st.sidebar.text(f"{key}: {value}")


def set_start_second(max_value: float):
    second = st.sidebar.slider(
        "start second", min_value=0, max_value=int(max_value), value=0, step=1)
    return second


def set_sampling_rate(current_value: int):
    options = [8000, 16000, 22050, 24000, 32000, 44100, 48000]
    index = options.index(current_value)
    sr = st.sidebar.selectbox(
        "Choose sampling rate", options=options, index=index)
    return sr
