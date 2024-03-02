import numpy as np
from classification_model.config.core import config
from classification_model.processing.features import ExtractLetterTransformer
from classification_model.processing.data_manager import data_preparation

def test_extract_letter_transformer(sample_input_data):
    # Given
    sample_input_data = data_preparation(dataframe=sample_input_data)
    transformer = ExtractLetterTransformer(
        variables=config.model_config.cabin_vars  # cabin
    )
    assert np.isnan(sample_input_data["cabin"].iat[9])
    assert sample_input_data["cabin"].iat[0] == "B5"

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert np.isnan(sample_input_data["cabin"].iat[9])
    assert subject["cabin"].iat[0] == "B"
