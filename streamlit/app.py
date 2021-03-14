# streamlit/app.py
# Streamlit application.

import streamlit as st

# Header
st.title("Tagifai · Applied ML · Made With ML")
"""by [Goku Mohandas](https://twitter.com/GokuMohandas)"""
st.info("🔍 &nbsp;Explore the different pages below:")

# Pages
# pages = {
#     "data": eda.main,
#     "baselines": baselines.main,
#     "experiments": experiments.main,
#     "best": best.main,
# }
# st.sidebar.header("Pages")
# selected_page = st.sidebar.radio("Select a page to explore:", list(pages.keys()))
# if selected_page in pages:
#     pages[selected_page]()
