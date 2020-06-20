import audioread

import streamlit as st

from pathlib import Path


@st.cache
def read_audio_bytes(path: Path):
    with open(path, "rb") as f:
        audio_bytes = f.read()

    return audio_bytes


@st.cache
def check_audio_info(path: Path):
    path_ = str(path)
    with audioread.audio_open(path_) as f:
        sr = f.samplerate
        ch = f.channels
        dur = f.duration

    return {"sample_rate": sr, "channels": ch, "duration": dur}


def display_media_audio(path: Path):
    format_ = path.name.split(".")[-1]
    if format_ == "mp3":
        format_ = "audio/mp3"
    elif format_ == "wav":
        format_ = "audio/wav"
    else:
        st.warning("Selected type is not readable format")

    if format_ in {"audio/wav", "audio/mp3"}:
        st.audio(read_audio_bytes(path), format=format_)


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
            return check_folder(str(new_folder))
