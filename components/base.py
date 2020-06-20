import streamlit as st

from pathlib import Path


def write_audio_info_to_sidebar(path: Path, info: dict):
    filename = path.name
    st.sidebar.subheader(f"Audio file: {filename}")
    st.sidebar.markdown("#### Basic info")

    for key, value in info.items():
        st.sidebar.text(f"{key}: {value}")
