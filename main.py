import streamlit as st

import components as C
import utils

if __name__ == "__main__":
    st.title("Audio Checking Tool")

    base_folder = st.text_input("specify directory which contains audio file")
    path = utils.check_folder(base_folder)

    if path is not None:
        audio_files = [
            f.name
            for f in (list(path.glob("*.wav")) + list(path.glob("*.mp3")))
        ]
        audio_file_name = st.selectbox(
            "Choose audio file", options=audio_files)

        audio_path = path / audio_file_name
        audio_info = utils.check_audio_info(audio_path)

        C.write_audio_info_to_sidebar(audio_path, audio_info)
        second = C.set_start_second(max_value=audio_info["duration"])
        sr = C.set_sampling_rate(audio_info["sample_rate"])
        utils.display_media_audio(audio_path, second)

        y = utils.read_audio(audio_path, audio_info, sr=sr)
        C.waveplot(y, sr)
        C.specshow(y, sr)
