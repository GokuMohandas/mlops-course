from pathlib import Path

import pandas as pd

import streamlit as st
from config import config
from tagifai import main, utils

# Title
st.title("MLOps Course · Made With ML")

# Sections
st.header("🔢 Data")
projects_fp = Path(config.DATA_DIR, "labeled_projects.csv")
df = pd.read_csv(projects_fp)
st.text(f"Projects (count: {len(df)})")
st.write(df)

st.header("📊 Performance")
performance_fp = Path(config.CONFIG_DIR, "performance.json")
performance = utils.load_dict(filepath=performance_fp)
st.text("Overall:")
st.write(performance["overall"])
tag = st.selectbox("Choose a tag: ", list(performance["class"].keys()))
st.write(performance["class"][tag])
tag = st.selectbox("Choose a slice: ", list(performance["slices"].keys()))
st.write(performance["slices"][tag])

st.header("🚀 Inference")
text = st.text_input("Enter text:", "Transfer learning with transformers for text classification.")
run_id = st.text_input("Enter run ID:", open(Path(config.CONFIG_DIR, "run_id.txt")).read())
prediction = main.predict_tag(text=text, run_id=run_id)
st.write(prediction)
