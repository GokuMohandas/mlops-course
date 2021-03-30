# streamlit/st_app.py
# Streamlit application.

import itertools
from collections import Counter, OrderedDict
from distutils.util import strtobool
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import STOPWORDS, WordCloud

import streamlit as st
from app import cli
from tagifai import config, data, eval, main, utils


@st.cache
def get_performance(model_dir):
    performance = utils.load_dict(filepath=Path(model_dir, "performance.json"))
    return performance


@st.cache
def get_diff(commit_a, commit_b):
    diff = cli.diff(commit_a=commit_a, commit_b=commit_b)
    return diff


@st.cache
def get_artifacts(model_dir):
    artifacts = main.load_artifacts(model_dir=model_dir)
    return artifacts


@st.cache
def evaluate_df(df, tags_dict, artifacts):
    # Retrieve artifacts
    params = artifacts["params"]

    # Prepare
    df, tags_above_freq, tags_below_freq = data.prepare(
        df=df,
        include=list(tags_dict.keys()),
        exclude=config.EXCLUDED_TAGS,
        min_tag_freq=int(params.min_tag_freq),
    )

    # Preprocess
    df.text = df.text.apply(
        data.preprocess,
        lower=bool(strtobool(str(params.lower))),
        stem=bool(strtobool(str(params.stem))),
    )

    # Evaluate
    y_true, y_pred, performance = eval.evaluate(df=df, artifacts=artifacts)
    sorted_tags = list(
        OrderedDict(
            sorted(performance["class"].items(), key=lambda tag: tag[1]["f1"], reverse=True)
        ).keys()
    )

    return y_true, y_pred, performance, sorted_tags, df


# Title
st.title("Tagifai · Applied ML · Made With ML")
"""by [Goku Mohandas](https://twitter.com/GokuMohandas)"""
st.info("🔍 Explore the different pages below.")

# Pages
pages = ["EDA", "Performance", "Inference", "Inspection"]
st.header("Pages")
selected_page = st.radio("Select a page:", pages, index=2)

if selected_page == "EDA":
    st.header("Exploratory Data Analysis")

    # Load data
    projects_fp = Path(config.DATA_DIR, "projects.json")
    tags_fp = Path(config.DATA_DIR, "tags.json")
    projects = utils.load_dict(filepath=projects_fp)
    tags_dict = utils.list_to_dict(utils.load_dict(filepath=tags_fp), key="tag")
    col1, col2 = st.beta_columns(2)
    with col1:
        st.subheader("Projects (sample)")
        st.write(projects[0])
    with col2:
        st.subheader("Tags")
        tag = st.selectbox("Choose a tag", list(tags_dict.keys()))
        st.write(tags_dict[tag])

    # Dataframe
    df = pd.DataFrame(projects)
    st.text(f"Projects (count: {len(df)}):")
    st.write(df)

    # Filter tags
    st.write("---")
    st.subheader("Filtering tags")
    st.write(
        "We want to determine what the minimum tag frequency is so that we have enough samples per tag for training."
    )
    min_tag_freq = st.slider("min_tag_freq", min_value=1, value=30, step=1)
    df, tags_above_freq, tags_below_freq = data.prepare(
        df=df,
        include=list(tags_dict.keys()),
        exclude=config.EXCLUDED_TAGS,
        min_tag_freq=min_tag_freq,
    )
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        st.write("**Most common tags**:")
        for item in tags_above_freq.most_common(5):
            st.write(item)
    with col2:
        st.write("**Tags that just made the cut**:")
        for item in tags_above_freq.most_common()[-5:]:
            st.write(item)
    with col3:
        st.write("**Tags that just missed the cut**:")
        for item in tags_below_freq.most_common(5):
            st.write(item)
    with st.beta_expander("Excluded tags"):
        st.write(config.EXCLUDED_TAGS)

    # Number of tags per project
    st.write("---")
    st.subheader("Plots")
    num_tags_per_project = [len(tags) for tags in df.tags]
    num_tags, num_projects = zip(*Counter(num_tags_per_project).items())
    plt.figure(figsize=(10, 3))
    ax = sns.barplot(list(num_tags), list(num_projects))
    plt.title("Tags per project", fontsize=20)
    plt.xlabel("Number of tags", fontsize=16)
    ax.set_xticklabels(range(1, len(num_tags) + 1), rotation=0, fontsize=16)
    plt.ylabel("Number of projects", fontsize=16)
    plt.show()
    st.pyplot(plt)

    # Distribution of tags
    tags = list(itertools.chain.from_iterable(df.tags.values))
    tags, tag_counts = zip(*Counter(tags).most_common())
    plt.figure(figsize=(25, 5))
    ax = sns.barplot(list(tags), list(tag_counts))
    plt.title("Tag distribution", fontsize=20)
    plt.xlabel("Tag", fontsize=16)
    ax.set_xticklabels(tags, rotation=90, fontsize=14)
    plt.ylabel("Number of projects", fontsize=16)
    plt.show()
    st.pyplot(plt)

    # Plot word clouds top top tags
    plt.figure(figsize=(20, 8))
    tag = st.selectbox("Choose a tag", tags, index=0)
    subset = df[df.tags.apply(lambda tags: tag in tags)]
    text = subset.text.values
    cloud = WordCloud(
        stopwords=STOPWORDS,
        background_color="black",
        collocations=False,
        width=500,
        height=300,
    ).generate(" ".join(text))
    plt.axis("off")
    plt.imshow(cloud)
    st.pyplot(plt)

    # Preprocessing
    st.write("---")
    st.subheader("Preprocessing")
    filters = st.text_input("filters", "[!\"'#$%&()*+,-./:;<=>?@\\[]^_`{|}~]")
    lower = st.checkbox("lower", True)
    stem = st.checkbox("stem", False)
    text = st.text_input("Input text", "Conditional generation using Variational Autoencoders.")
    preprocessed_text = data.preprocess(text=text, lower=lower, stem=stem, filters=filters)
    st.text("Preprocessed text")
    st.write(preprocessed_text)

elif selected_page == "Performance":
    st.header("Performance")

    # Best runs performance
    st.subheader("Current model")
    performance = get_performance(model_dir=config.MODEL_DIR)
    diff = get_diff(commit_a="workspace", commit_b="head")
    with st.beta_expander("Overall performance", expanded=True):
        st.json(performance["overall"])
    with st.beta_expander("Per-class performance"):
        st.json(performance["class"])
    with st.beta_expander("Slices performance"):
        st.json(performance["slices"])
    with st.beta_expander("Behavioral report"):
        st.json(performance["behavioral"])
    with st.beta_expander("Hyperparameter changes"):
        st.text("Diff comparing workspace to currently deployed (HEAD)")
        st.json(diff["params"])
    with st.beta_expander("Improvements"):
        st.text("Improvements comparing workspace to currently deployed (HEAD)")
        st.json(diff["metrics"]["improvements"])
    with st.beta_expander("Regressions"):
        st.text("Regressions comparing workspace to currently deployed (HEAD)")
        st.json(diff["metrics"]["regressions"])

elif selected_page == "Inference":
    st.header("Inference")
    text = st.text_input(
        "Enter text:",
        "Transfer learning with transformers for self-supervised learning.",
    )
    prediction = cli.predict_tags(text=text)
    st.text("Prediction:")
    st.write(prediction)

elif selected_page == "Inspection":
    st.header("Inspection")
    st.write(
        "We're going to inspect the TP, FP and FN samples across our different tags. It's a great way to catch issues with labeling (FP), weaknesses (FN), etc."
    )
    st.info(
        "Ideally we would connect an inspection system like this with our annotation pipelines so any changes can be reviewed and incorporated."
    )

    # Load data
    projects_fp = Path(config.DATA_DIR, "projects.json")
    tags_fp = Path(config.DATA_DIR, "tags.json")
    projects = utils.load_dict(filepath=projects_fp)
    tags_dict = utils.list_to_dict(utils.load_dict(filepath=tags_fp), key="tag")
    df = pd.DataFrame(projects)

    # Get performance
    artifacts = get_artifacts(model_dir=config.MODEL_DIR)
    label_encoder = artifacts["label_encoder"]
    y_true, y_pred, performance, sorted_tags, df = evaluate_df(
        df=df,
        tags_dict=tags_dict,
        artifacts=artifacts,
    )
    tag = st.selectbox("Choose a tag", sorted_tags, index=0)
    st.json(performance["class"][tag])

    # TP, FP, FN samples
    index = label_encoder.class_to_index[tag]
    tp, fp, fn = [], [], []
    for i in range(len(y_true)):
        true = y_true[i][index]
        pred = y_pred[i][index]
        if true and pred:
            tp.append(i)
        elif not true and pred:
            fp.append(i)
        elif true and not pred:
            fn.append(i)

    # Samples
    num_samples = 3
    with st.beta_expander("True positives"):
        if len(tp):
            for i in tp[:num_samples]:
                st.write(f"{df.text.iloc[i]}")
                st.text("True")
                st.write(label_encoder.decode([y_true[i]])[0])
                st.text("Predicted")
                st.write(label_encoder.decode([y_pred[i]])[0])
    with st.beta_expander("False positives"):
        if len(fp):
            for i in fp[:num_samples]:
                st.write(f"{df.text.iloc[i]}")
                st.text("True")
                st.write(label_encoder.decode([y_true[i]])[0])
                st.text("Predicted")
                st.write(label_encoder.decode([y_pred[i]])[0])
    with st.beta_expander("False negatives"):
        if len(fn):
            for i in fn[:num_samples]:
                st.write(f"{df.text.iloc[i]}")
                st.text("True")
                st.write(label_encoder.decode([y_true[i]])[0])
                st.text("Predicted")
                st.write(label_encoder.decode([y_pred[i]])[0])
    st.write("\n")
    st.warning("Looking at false positives is a great way to find mislabeled data.")

else:
    st.text("Please select a valid page option from above...")

st.write("---")

# Resources
"""
## Resources

- 🎓 Lessons: https://madewithml.com/
- 🐙 Repository: https://github.com/GokuMohandas/applied-ml
- 📘 Documentation: https://gokumohandas.github.io/applied-ml/
- 📬 Subscribe: https://newsletter.madewithml.com
"""
