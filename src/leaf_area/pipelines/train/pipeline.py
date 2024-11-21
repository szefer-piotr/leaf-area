"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import torch_device_test


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            name = "torch_device_test",
            inputs=None,
            outputs="device",
            func=torch_device_test
        )
    ])
