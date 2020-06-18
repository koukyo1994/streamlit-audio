import streamlit as st

import utils

if __name__ == "__main__":
    st.title("Audio Checking Tool")

    uploaded_file = st.file_uploader("Choose a master csv file", type=str)

    if uploaded_file is not None:
        df = utils.read_csv(uploaded_file)
        nrows, ncols = df.shape
        st.write(f"Number of rows: {nrows}, Number of columns: {ncols}")

        analysis_option = st.selectbox(
            "Which action would you like to do?",
            options=["-", "Check audio", "Check augmentation", "Plot"])

        if analysis_option == "Check audio":
            pass
