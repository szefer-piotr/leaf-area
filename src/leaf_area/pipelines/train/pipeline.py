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
    build_model_dictionary,
    instantiate_model,
    initialize_training
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
            name='build_model_dictionary_node',
            inputs='model_dict',
            outputs='callable_model_dict',
            func=build_model_dictionary,
        ),
        node(
            name='initialize_model_node',
            inputs=['callable_model_dict',
                    'params:model_definition_params',
                    'params:final_layer_params',
                    'params:device'],
            outputs='model',
            func=instantiate_model
        ),
        node(
            name='initialize_training_node',
            inputs=['model', "params:initializer_configs"],
            outputs="initializers",
            func=initialize_training,
        )
    ])
