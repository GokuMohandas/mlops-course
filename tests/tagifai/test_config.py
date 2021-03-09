# test_config.py
# Test tagifai/config.py components.

from tagifai import config


def test_config():
    assert config.logger.name == "root"
