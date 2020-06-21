import audioread
import io
import tempfile
import struct
import wave

import librosa
import numpy as np
import streamlit as st

from pathlib import Path
from typing import Optional


@st.cache
def read_audio(path: Path, info: dict, sr: Optional[int] = None):
    if sr is None:
        sr = info["sample_rate"]
    y, _ = librosa.load(path, sr=sr, mono=True, res_type="kaiser_fast")
    return y


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


def display_media_audio_from_ndarray(y: np.ndarray, sr: int):
    max_num = 32767.0 / y.max()
    y_hex = (y * max_num).astype(np.int16)
    binary_wave = struct.pack("h" * len(y_hex), *(y_hex.tolist()))

    with tempfile.TemporaryFile() as fp:
        w = wave.Wave_write(fp)  # type: ignore
        params = (1, 2, sr, len(binary_wave), "NONE", "not compressed")
        w.setparams(params)  # type: ignore
        w.writeframes(binary_wave)

        fp.seek(0)
        bytesio = io.BytesIO(fp.read())

    st.audio(bytesio)


def display_media_audio(path: Path, start_second: int = 0):
    format_ = path.name.split(".")[-1]
    if format_ == "mp3":
        format_ = "audio/mp3"
    elif format_ == "wav":
        format_ = "audio/wav"
    else:
        st.warning("Selected type is not readable format")

    if format_ in {"audio/wav", "audio/mp3"}:
        st.audio(
            read_audio_bytes(path), start_time=start_second, format=format_)


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
