# test_data.py
# Test tagifai/data.py components.

import itertools
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tagifai import config, data, utils


@pytest.fixture
def tags():
    tags_fp = Path(config.DATA_DIR, "tags.json")
    tags_dict = utils.list_to_dict(utils.load_dict(filepath=tags_fp), key="tag")
    tags = list(tags_dict.keys())
    return tags


@pytest.fixture
def df():
    projects_fp = Path(config.DATA_DIR, "projects.json")
    projects_dict = utils.load_dict(filepath=projects_fp)
    df = pd.DataFrame(projects_dict)
    return df


@pytest.mark.parametrize(
    "items, include, filtered",
    [
        # one item
        (["apple"], ["apple"], ["apple"]),
        # multiple items
        (
            ["apple", "banana", "grape", "orange"],
            ["apple"],
            ["apple"],
        ),
        # multiple include
        (
            ["apple", "banana", "grape", "orange"],
            ["apple", "grape"],
            ["apple", "grape"],
        ),
        # no include
        (
            ["apple", "banana", "grape", "orange"],
            [],
            [],
        ),
    ],
)
def test_filter_items(items, include, filtered):
    assert data.filter_items(items=items, include=include) == filtered


def test_clean(tags, df):
    min_tag_freq = 30
    df, tags_above_frequency = data.clean(
        df=df,
        include=tags,
        exclude=config.EXCLUDE,
        min_tag_freq=min_tag_freq,
    )
    all_tags = list(itertools.chain.from_iterable(df.tags))
    assert Counter(all_tags).most_common()[-1][1] >= min_tag_freq


@pytest.mark.parametrize(
    "text, lower, stem, filters, stopwords, preprocessed_text",
    [
        ("Hello worlds", False, False, "", [], "Hello worlds"),
        ("Hello worlds", True, False, "", [], "hello worlds"),
        ("Hello worlds", False, True, "", [], "Hello world"),
        ("Hello worlds", True, True, "", [], "hello world"),
        ("Hello worlds", True, True, "l", [], "heo word"),
        ("Hello worlds", True, True, "", ["world"], "hello world"),
        ("Hello worlds", True, True, "", ["worlds"], "hello"),
    ],
)
def test_preprocess(text, lower, stem, filters, stopwords, preprocessed_text):
    assert (
        data.preprocess(
            text=text,
            lower=lower,
            stem=stem,
            filters=filters,
            stopwords=stopwords,
        )
        == preprocessed_text
    )


class TestLabelEncoder(object):
    @classmethod
    def setup_class(cls):
        """Called before every class initialization."""
        pass

    @classmethod
    def teardown_class(cls):
        """Called after every class initialization."""
        pass

    def setup_method(self):
        """Called before every method."""
        self.label_encoder = data.LabelEncoder()

    def teardown_method(self):
        """Called after every method."""
        del self.label_encoder

    def test_empty_init(self):
        label_encoder = data.LabelEncoder()
        assert label_encoder.index_to_class == {}
        assert len(label_encoder.classes) == 0

    def test_dict_init(self):
        class_to_index = {"apple": 0, "banana": 1}
        label_encoder = data.LabelEncoder(class_to_index=class_to_index)
        assert label_encoder.index_to_class == {0: "apple", 1: "banana"}
        assert len(label_encoder.classes) == 2

    def test_len(self):
        assert len(self.label_encoder) == 0

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as dp:
            fp = Path(dp, "label_encoder.json")
            self.label_encoder.save(fp=fp)
            label_encoder = data.LabelEncoder.load(fp=fp)
            assert len(label_encoder.classes) == 0

    @pytest.mark.parametrize(
        "label_encoder, output",
        [
            (data.MultiClassLabelEncoder(), "<MultiClassLabelEncoder(num_classes=0)>"),
            (data.MultiLabelLabelEncoder(), "<MultiLabelLabelEncoder(num_classes=0)>"),
        ],
    )
    def test_str(self, label_encoder, output):
        assert str(label_encoder) == output

    @pytest.mark.parametrize(
        "label_encoder, y",
        [
            (data.MultiClassLabelEncoder(), ["apple", "apple", "banana"]),
            (data.MultiLabelLabelEncoder(), [["apple"], ["apple", "banana"]]),
        ],
    )
    def test_fit(self, label_encoder, y):
        label_encoder.fit(y)
        assert "apple" in label_encoder.class_to_index
        assert "banana" in label_encoder.class_to_index
        assert len(label_encoder.classes) == 2

    @pytest.mark.parametrize(
        "label_encoder, y, y_encoded",
        [
            (
                data.MultiClassLabelEncoder(class_to_index={"apple": 0, "banana": 1}),
                ["apple", "apple", "banana"],
                [0, 0, 1],
            ),
            (
                data.MultiLabelLabelEncoder(class_to_index={"apple": 0, "banana": 1}),
                [["apple"], ["apple", "banana"]],
                [[1, 0], [1, 1]],
            ),
        ],
    )
    def test_encode_decode(self, label_encoder, y, y_encoded):
        label_encoder.fit(y)
        assert np.array_equal(label_encoder.encode(y), np.array(y_encoded))
        assert label_encoder.decode(y_encoded) == y


def test_iterative_train_test_split(tags, df):
    # Process
    df, tags_above_frequency = data.clean(df=df, include=tags, min_tag_freq=1)
    df.text = df.text.apply(data.preprocess)

    # Encode labels
    labels = df.tags
    label_encoder = data.MultiLabelLabelEncoder()
    label_encoder.fit(labels)
    y = label_encoder.encode(labels)

    # Split data
    X = df.text.to_numpy()
    X_train, X_, y_train, y_ = data.iterative_train_test_split(X=X, y=y, train_size=0.7)
    X_val, X_test, y_val, y_test = data.iterative_train_test_split(X=X_, y=y_, train_size=0.5)

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    assert len(X_train) / float(len(X)) == pytest.approx(0.7, abs=0.05)  # 0.7 ± 0.05
    assert len(X_val) / float(len(X)) == pytest.approx(0.15, abs=0.05)  # 0.15 ± 0.05
    assert len(X_test) / float(len(X)) == pytest.approx(0.15, abs=0.05)  # 0.15 ± 0.05


class TestTokenizer(object):
    def setup_method(self):
        """Called before every method."""
        self.tokenizer = data.Tokenizer(char_level=True, num_tokens=None)

    def teardown_method(self):
        """Called after every method."""
        del self.tokenizer

    @pytest.mark.parametrize(
        "char_level, num_tokens, separator, token_to_index, expected_token_to_index",
        [
            (True, None, "", None, {"<PAD>": 0, "<UNK>": 1}),
            (False, None, " ", None, {"<PAD>": 0, "<UNK>": 1}),
            (
                False,
                None,
                " ",
                {"<PAD>": 0, "<UNK>": 1, "hello": 2},
                {"<PAD>": 0, "<UNK>": 1, "hello": 2},
            ),
        ],
    )
    def test_init(self, char_level, num_tokens, separator, token_to_index, expected_token_to_index):
        tokenizer = data.Tokenizer(
            char_level=char_level, num_tokens=num_tokens, token_to_index=token_to_index
        )
        assert tokenizer.separator == separator
        assert tokenizer.token_to_index == expected_token_to_index

    def test_len(self):
        assert len(self.tokenizer) == 2

    def test_str(self):
        assert str(self.tokenizer) == f"<Tokenizer(num_tokens={len(self.tokenizer)})>"

    @pytest.mark.parametrize(
        "char_level, num_tokens, texts, vocab_size",
        [(False, None, ["hello world", "goodbye"], 5), (False, 4, ["hello world", "goodbye"], 4)],
    )
    def test_fit_on_texts(self, char_level, num_tokens, texts, vocab_size):
        tokenizer = data.Tokenizer(char_level=char_level, num_tokens=num_tokens)
        tokenizer.fit_on_texts(texts=texts)
        assert len(tokenizer) == vocab_size

    @pytest.mark.parametrize(
        "tokenizer, texts, sequences, decoded",
        [
            (
                data.Tokenizer(
                    char_level=False,
                    token_to_index={"<PAD>": 0, "<UNK>": 1, "hello": 2, "world": 3},
                ),
                ["hello world", "hi world", "apple"],
                [[2, 3], [1, 3], [1]],
                ["hello world", "<UNK> world", "<UNK>"],
            ),
            (
                data.Tokenizer(
                    char_level=True, token_to_index={"<PAD>": 0, "<UNK>": 1, " ": 2, "a": 3, "b": 4}
                ),
                ["ab", "b", "a x ab"],
                [[3, 4], [4], [3, 2, 1, 2, 3, 4]],
                ["ab", "b", "a <UNK> ab"],
            ),
        ],
    )
    def test_encode_decode(self, tokenizer, texts, sequences, decoded):
        assert tokenizer.texts_to_sequences(texts=texts) == sequences
        assert tokenizer.sequences_to_texts(sequences=sequences) == decoded

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as dp:
            tokenizer = data.Tokenizer(
                char_level=False, token_to_index={"<PAD>": 0, "<UNK>": 1, "hello": 2, "world": 3}
            )
            fp = Path(dp, "label_encoder.json")
            tokenizer.save(fp=fp)
            tokenizer = data.Tokenizer.load(fp=fp)
            assert len(tokenizer) == 4


def test_pad_sequences():
    # Explicit max len
    seq = np.array([[1, 2, 3], [1, 2]], dtype=object)
    padded_seq = data.pad_sequences(sequences=seq, max_seq_len=5)
    assert np.array_equal(padded_seq, np.array([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]]))

    # Implicit max len
    seq = np.array([[1, 2, 3], [1, 2]], dtype=object)
    padded_seq = data.pad_sequences(sequences=seq)
    assert np.array_equal(padded_seq, np.array([[1, 2, 3], [1, 2, 0]]))


class TestCNNTextDataset(object):
    def setup_method(self):
        """Called before every method."""
        self.X = [[4, 2, 3, 0], [2, 4, 3, 3], [2, 3, 0, 0]]
        self.y = [[0, 1], [1, 1], [1, 0]]
        self.max_filter_size = 2
        self.batch_size = 1
        self.dataset = data.CNNTextDataset(X=self.X, y=self.y, max_filter_size=self.max_filter_size)

    def teardown_method(self):
        """Called after every method."""
        del self.dataset

    def test_init(self):
        assert self.max_filter_size == self.dataset.max_filter_size

    def test_len(self):
        assert len(self.X) == len(self.dataset)

    def test_str(self):
        assert str(self.dataset) == f"<Dataset(N={len(self.dataset)})>"

    def test_get_item(self):
        assert self.dataset[0] == [self.X[0], self.y[0]]
        assert self.dataset[-1] == [self.X[-1], self.y[-1]]

    @pytest.mark.parametrize(
        "batch_size, drop_last, num_batches",
        [(1, False, 3), (2, False, 2), (2, True, 1), (3, False, 1)],
    )
    def test_create_dataloader(self, batch_size, drop_last, num_batches):
        dataloader = self.dataset.create_dataloader(batch_size=batch_size, drop_last=drop_last)
        assert len(dataloader) == num_batches

    def test_dataloader(self):
        batch_size = 2
        dataloader = self.dataset.create_dataloader(batch_size=batch_size, drop_last=False)
        max_seq_len = max(self.max_filter_size, max(len(sequence) for sequence in self.X))
        for batch in dataloader:
            assert len(batch) <= batch_size
            assert np.shape(batch[0])[-1] == max_seq_len
