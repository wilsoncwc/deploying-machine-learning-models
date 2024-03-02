import pytest

from classification_model.config.core import config
from classification_model.processing.data_manager import load_raw_dataset

from sklearn.model_selection import train_test_split

@pytest.fixture()
def sample_input_data():
    data = load_raw_dataset()
    return data
