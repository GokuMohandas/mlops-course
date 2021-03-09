# custom_dataset.py
# Custom datasets for Great Expectations.

import great_expectations as ge


class CustomPandasDataset(ge.dataset.PandasDataset):
    _data_asset_type = "CustomPandasDataset"

    @ge.dataset.MetaPandasDataset.column_map_expectation
    def expect_column_list_values_to_be_not_null(self, column):
        return column.map(lambda x: None not in x)

    @ge.dataset.MetaPandasDataset.column_map_expectation
    def expect_column_list_values_to_be_unique(self, column):
        return column.map(lambda x: len(x) == len(set(x)))
