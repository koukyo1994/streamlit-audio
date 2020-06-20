import pandas as pd
import streamlit as st

from pathlib import Path


@st.cache
def read_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df


@st.cache
def read_audio(uploaded_file):
    print(dir(uploaded_file))


def check_folder(folder: str):
    path = Path(folder)
    if not path.exists():
        st.warning("specified folder does not exist")
        return
    else:
        wavs = list(path.glob("*.wav"))
        mp3s = list(path.glob("*.mp3"))
        subdirs = [
            subpaths for subpaths in path.glob("*") if subpaths.is_dir()
        ]

        if len(wavs) > 0:
            st.success(f"Found {len(wavs)} wav files")
            return path
        if len(mp3s) > 0:
            st.success(f"Found {len(mp3s)} mp3 files")
            return path
        if len(subdirs) == 0:
            st.warning("No wav or mp3 found under the directory you specified")
            return
        else:
            subdir_names = [subdir.name for subdir in subdirs]
            subfolder = st.selectbox(
                f"Pick one folder below {str(folder)}",
                options=subdir_names,
                key=f"{str(folder)}")

            new_folder = path / subfolder
            check_folder(str(new_folder))
