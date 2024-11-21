"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

import leaf_area.pipelines.preprocess as prep
import leaf_area.pipelines.train as train

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    data_processing_pipeline = prep.create_pipeline()
    model_training_pipeline = train.create_pipeline()

    return {
        '__default__': data_processing_pipeline + model_training_pipeline,
        'data_processing_pipeline': data_processing_pipeline,
        'model_training_pipeline': model_training_pipeline,
    }
