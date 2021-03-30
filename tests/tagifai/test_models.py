# tests/tagifai/test_models.py
# Test tagifai/models.py components.

from argparse import Namespace

import torch

from tagifai import models, utils


class TestCNN:
    def setup_method(self):
        """Called before every method."""
        # Params
        self.max_filter_size = 4
        self.embedding_dim = 128
        self.num_filters = 100
        self.hidden_dim = 128
        self.dropout_p = 0.5
        params = Namespace(
            max_filter_size=self.max_filter_size,
            embedding_dim=self.embedding_dim,
            num_filters=self.num_filters,
            hidden_dim=self.hidden_dim,
            dropout_p=self.dropout_p,
        )

        # Model
        self.vocab_size = 1000
        self.num_classes = 10
        utils.set_seed()
        self.cnn = models.initialize_model(
            params=params, vocab_size=self.vocab_size, num_classes=self.num_classes
        )

    def teardown_method(self):
        """Called after every method."""
        del self.cnn

    def test_initialize_model(self):
        utils.set_seed()
        model = models.CNN(
            embedding_dim=self.embedding_dim,
            vocab_size=self.vocab_size,
            num_filters=self.num_filters,
            filter_sizes=[1, 2, 3, 4],
            hidden_dim=self.hidden_dim,
            dropout_p=self.dropout_p,
            num_classes=self.num_classes,
        )
        for param1, param2 in zip(self.cnn.parameters(), model.parameters()):
            assert not param1.data.ne(param2.data).sum() > 0
        assert self.cnn.filter_sizes == model.filter_sizes

    def test_init(self):
        assert self.cnn.embeddings.weight.shape == (self.vocab_size, self.embedding_dim)

    def test_forward(self):
        x = torch.LongTensor([[2, 3, 0], [1, 3, 2]])
        z = self.cnn.forward(inputs=[x])
        assert z.shape == (len(x), self.num_classes)
