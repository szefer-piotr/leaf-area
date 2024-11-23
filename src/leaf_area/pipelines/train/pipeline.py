"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    torch_device_test,
    load_data,
    split_data,
    create_transformations,
    create_datasets,
    instantiate_model
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            name = 'torch_device_test',
            inputs=None,
            outputs='device',
            func=torch_device_test
        ),
        node(
            name = 'split_data_node',
            inputs=['data_filtered'],
            outputs=['X_train', 'X_val', 'X_test'],
            func = split_data
        ),
        node(
            name='create_transformations_node',
            inputs=None,
            outputs='data_transformations',
            func=create_transformations
        ),
        node(
            name='create_datasets_node',
            inputs=['X_train', 
                    'X_val', 
                    'X_test', 
                    'data_transformations',
                    'params:target_transformations'],
            outputs=['image_datasets', 'dataset_sizes'],
            func=create_datasets
        ),
        node(
            name='initialize_model_node',
            inputs=['models_dict',
                    'params:model_definition_params',
                    'params:final_layer_params',
                    'params:device'],
            outputs='model',
            func=instantiate_model
        ),
    ])
