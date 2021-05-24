# This is an example feature definition file

from datetime import datetime
from pathlib import Path

from feast import Entity, Feature, FeatureView, ValueType
from feast.data_source import FileSource
from google.protobuf.duration_pb2 import Duration

from tagifai import config

# Read data
START_TIME = "2020-02-17"
project_details = FileSource(
    path=str(Path(config.DATA_DIR, "projects.parquet")),
    event_timestamp_column="created_on",
)

# Define an entity for the project
project = Entity(
    name="id",
    value_type=ValueType.INT64,
    description="project id",
)

# Define a Feature View for each project
# Will be used for online serving
project_details_view = FeatureView(
    name="project_details",
    entities=["id"],
    ttl=Duration(
        seconds=(datetime.today() - datetime.strptime(START_TIME, "%Y-%m-%d")).days * 24 * 60 * 60
    ),
    features=[
        Feature(name="title", dtype=ValueType.STRING),
        Feature(name="description", dtype=ValueType.STRING),
        Feature(name="tags", dtype=ValueType.STRING_LIST),
    ],
    online=True,
    input=project_details,
    tags={},
)
