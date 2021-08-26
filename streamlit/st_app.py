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
from app import cli, config
from tagifai import data, eval, main, utils


@st.cache
def get_performance(model_dir):
    performance = utils.load_dict(filepath=Path(model_dir, "performance.json"))
    return performance


@st.cache
def get_tags(author=config.AUTHOR, repo=config.REPO):
    # Get list of tags
    tags_list = ["workspace"] + [
        tag["name"]
        for tag in utils.load_json_from_url(
            url=f"https://api.github.com/repos/{author}/{repo}/tags"
        )
    ]

    # Get metadata by tag
    tags = {}
    for tag in tags_list:
        tags[tag] = {}
        tags[tag]["params"] = cli.params(tag=tag, verbose=False)
        tags[tag]["performance"] = pd.json_normalize(
            cli.performance(tag=tag, verbose=False), sep="."
        ).to_dict(orient="records")[0]

    return tags


@st.cache
def get_diff(author=config.AUTHOR, repo=config.REPO, tag_a="workspace", tag_b=""):
    params_diff, performance_diff = cli.diff(author=author, repo=repo, tag_a=tag_a, tag_b=tag_b)
    return params_diff, performance_diff


@st.cache
def get_artifacts(run_id):
    artifacts = main.load_artifacts(run_id=run_id)
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
st.title("Tagifai · MLOps · Made With ML")
"""by [Goku Mohandas](https://twitter.com/GokuMohandas)"""
st.info("🔍 Explore the different pages below.")

# Pages
pages = ["Data", "Performance", "Inference", "Inspection"]
st.header("Pages")
selected_page = st.radio("Select a page:", pages, index=2)

if selected_page == "Data":
    st.header("Data")

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
    st.subheader("Annotation")
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
    st.subheader("Exploratory Data Analysis")
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
    plt.figure(figsize=(10, 3))
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
    text = st.text_input("Input text", "Conditional generation using Variational Autoencoders.")
    filters = st.text_input("filters", "[!\"'#$%&()*+,-./:;<=>?@\\[]^_`{|}~]")
    lower = st.checkbox("lower", True)
    stem = st.checkbox("stem", False)
    preprocessed_text = data.preprocess(text=text, lower=lower, stem=stem, filters=filters)
    st.text("Preprocessed text:")
    st.write(preprocessed_text)

elif selected_page == "Performance":
    st.header("Performance")

    # Get tags and respective parameters and performance
    tags = get_tags(author=config.AUTHOR, repo=config.REPO)

    # Key metrics
    key_metrics = [
        "overall.f1",
        "overall.precision",
        "overall.recall",
        "behavioral.score",
        "slices.overall.f1",
        "slices.overall.precision",
        "slices.overall.recall",
    ]

    # Key metric values over time
    key_metrics_over_time = {}
    for metric in key_metrics:
        key_metrics_over_time[metric] = {}
        for tag in tags:
            key_metrics_over_time[metric][tag] = tags[tag]["performance"][metric]
    st.line_chart(key_metrics_over_time)

    # Compare two performance
    st.subheader("Compare performances:")
    d = {}
    col1, col2 = st.beta_columns(2)
    with col1:
        tag_a = st.selectbox("Tag A", list(tags.keys()), index=0)
        d[tag_a] = {"links": {}}
    with col2:
        tag_b = st.selectbox("Tag B", list(tags.keys()), index=1)
        d[tag_b] = {"links": {}}
    if tag_a == tag_b:
        raise Exception("Tags must be different in order to compare them.")

    # Diffs
    params_diff, performance_diff = get_diff(
        author=config.AUTHOR, repo=config.REPO, tag_a=tag_a, tag_b=tag_b
    )
    with st.beta_expander("Key metrics", expanded=True):
        key_metrics_dict = {metric: performance_diff[metric] for metric in key_metrics}
        key_metrics_diffs = [key_metrics_dict[metric]["diff"] * 100 for metric in key_metrics_dict]
        plt.figure(figsize=(10, 5))
        ax = sns.barplot(
            x=key_metrics,
            y=key_metrics_diffs,
            palette=["green" if value >= 0 else "red" for value in key_metrics_diffs],
        )
        ax.axhline(0, ls="--")
        for i, (metric, value) in enumerate(zip(key_metrics, key_metrics_diffs)):
            ax.annotate(
                s=f"{value:.2f}%\n({key_metrics_dict[metric][tag_a]:.2f} / {key_metrics_dict[metric][tag_b]:.2f})\n\n",
                xy=(i, value),
                ha="center",
                va="center",
                xytext=(0, 10),
                textcoords="offset points",
                fontsize=12,
            )
        sns.despine(ax=ax, bottom=False, left=False)
        plt.xlabel("Metric", fontsize=16)
        ax.set_xticklabels(key_metrics, rotation=30, fontsize=10)
        plt.ylabel("Diff (%)", fontsize=16)
        plt.show()
        st.pyplot(plt)

    with st.beta_expander("Hyperparameters"):
        st.json(params_diff)
    with st.beta_expander("Improvements"):
        st.json({metric: value for metric, value in performance_diff.items() if value["diff"] >= 0})
    with st.beta_expander("Regressions"):
        st.json({metric: value for metric, value in performance_diff.items() if value["diff"] < 0})

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

    # Load data
    projects_fp = Path(config.DATA_DIR, "projects.json")
    tags_fp = Path(config.DATA_DIR, "tags.json")
    projects = utils.load_dict(filepath=projects_fp)
    tags_dict = utils.list_to_dict(utils.load_dict(filepath=tags_fp), key="tag")
    df = pd.DataFrame(projects)

    # Get performance
    run_id = open(Path(config.MODEL_DIR, "run_id.txt")).read()
    artifacts = get_artifacts(run_id=run_id)
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
    st.warning(
        "Be careful not to make decisions based on predicted probabilities before [calibrating](https://arxiv.org/abs/1706.04599) them to reliably use as measures of confidence."
    )
    """
    ### Extensions

    - Use false positives to identify potentially mislabeled data.
    - Connect inspection pipelines with annotation systems so that changes to the data can be reviewed and incorporated.
    - Inspect FP / FN samples by [estimating training data influences (TracIn)](https://arxiv.org/abs/2002.08484) on their predictions.
    - Inspect the trained model's behavior under various conditions using the [WhatIf](https://pair-code.github.io/what-if-tool/) tool.
    """


else:
    st.text("Please select a valid page option from above...")

st.write("---")

# Resources
"""
## Resources

- 🎓 Lessons: https://madewithml.com/
- 🐙 Repository: https://github.com/GokuMohandas/MLOps
- 📘 Documentation: https://gokumohandas.github.io/mlops/
- 📬 Subscribe: https://newsletter.madewithml.com
"""
