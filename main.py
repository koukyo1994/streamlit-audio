import pandas as pd
import streamlit as st

import components as C
import utils

if __name__ == "__main__":
    st.title("Audio Checking Tool")

    base_folder = st.text_input("specify directory which contains audio file")
    tp = pd.read_csv("../input/train_tp.csv")
    fp = pd.read_csv("../input/train_fp.csv")
    st.dataframe(tp)
    path = utils.check_folder(base_folder)
    if path is not None:
        audio_files = sorted([
            f.name
            for f in (list(path.glob("*.wav")) + list(path.glob("*.mp3")) +
                      list(path.glob("*.flac")))
        ])
        audio_file_name = st.selectbox(
            "Choose audio file", options=audio_files)
        audio_id = audio_file_name.replace(".flac", "")
        tp_in_audio = tp.query(f"recording_id == '{audio_id}'").reset_index()
        fp_in_audio = fp.query(f"recording_id == '{audio_id}'").reset_index()

        st.text("tp")
        st.dataframe(tp_in_audio)
        st.text("fp")
        st.dataframe(fp_in_audio)

        audio_path = path / audio_file_name
        audio_info = utils.check_audio_info(audio_path)

        C.write_audio_info_to_sidebar(audio_path, audio_info)
        second = C.set_start_second(max_value=audio_info["duration"])
        sr = C.set_sampling_rate(audio_info["sample_rate"])

        options = st.sidebar.selectbox(
            "Audio option",
            options=["normal", "preprocessing", "augmentations"])
        utils.display_media_audio(audio_path, second)

        annotation = st.sidebar.file_uploader(
            "Upload annotation file if exist")
        if annotation is not None:
            event_level_annotation = utils.read_csv(annotation)
        else:
            event_level_annotation = None

        y = utils.read_audio(audio_path, audio_info, sr=sr)
        if options == "preprocessing":
            y_processed = C.preprocess_on_wave(
                y, sr=sr, audio_path=str(audio_path))
            if y_processed is not None:
                st.text("Processed audio")
                utils.display_media_audio_from_ndarray(y_processed, sr)
                if event_level_annotation is None:
                    C.waveplot(y, sr, y_processed)
                    C.specshow(y, sr, y_processed)
                else:
                    C.waveplot_with_annotation(y, sr, event_level_annotation,
                                               audio_file_name, y_processed)
                    C.specshow_with_annotation(y, sr, event_level_annotation,
                                               audio_file_name, y_processed)
        elif options == "augmentations":
            y_processed = C.augmentations_on_wave(
                y, sr=sr)
            if y_processed is not None:
                st.text("Processed audio")
                utils.display_media_audio_from_ndarray(y_processed, sr)
                if event_level_annotation is None:
                    C.waveplot(y, sr, y_processed)
                    C.specshow(y, sr, y_processed)
                else:
                    C.waveplot_with_annotation(y, sr, event_level_annotation,
                                               audio_file_name, y_processed)
                    C.specshow_with_annotation(y, sr, event_level_annotation,
                                               audio_file_name, y_processed)
        else:
            if event_level_annotation is None:
                C.waveplot(y, sr, tp=tp_in_audio, fp=fp_in_audio)
                C.specshow(y, sr, tp=tp_in_audio, fp=fp_in_audio)
            else:
                C.waveplot_with_annotation(
                    y,
                    sr,
                    event_level_annotation,
                    audio_file_name,
                    processed=None)
                C.specshow_with_annotation(
                    y,
                    sr,
                    event_level_annotation,
                    audio_file_name,
                    y_processed=None)
