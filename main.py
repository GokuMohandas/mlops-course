# features/main.py
# Accessing features via offline/online stores
# for training/inference, respectively

from datetime import datetime

import pandas as pd
from feast import FeatureStore

# Identify entities you want to pull data for
entity_df = pd.DataFrame.from_dict({"id": [1], "event_timestamp": [datetime(2021, 5, 1, 0, 0, 0)]})

print(entity_df.head())

# Get historical features
store = FeatureStore(repo_path="features")
training_df = store.get_historical_features(
    entity_df=entity_df,
    feature_refs=["project_details:title", "project_details:description", "project_details:tags"],
).to_df()

print(training_df.head())


# Identify entities you want to pull data for
entity_df = pd.DataFrame.from_dict({"id": [1], "event_timestamp": [datetime(2019, 5, 1, 0, 0, 0)]})

print(entity_df.head())

# Get historical features
store = FeatureStore(repo_path="features")
training_df = store.get_historical_features(
    entity_df=entity_df,
    feature_refs=["project_details:title", "project_details:description", "project_details:tags"],
).to_df()

print(training_df.head())


store = FeatureStore(repo_path="features")
feature_vector = store.get_online_features(
    feature_refs=["project_details:title", "project_details:description", "project_details:tags"],
    entity_rows=[{"id": 1}],
).to_dict()

print(feature_vector)
