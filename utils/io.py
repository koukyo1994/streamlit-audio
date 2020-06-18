import pandas as pd
import streamlit as st


@st.cache
def read_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df
