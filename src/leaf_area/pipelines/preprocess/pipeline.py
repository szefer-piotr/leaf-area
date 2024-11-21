"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    organize_images_and_tables,
    concatenate_tabular_data
)



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            name='organize_image_and_table_files_node',
            inputs=['params:data_source_path', 'params:images_destination_path', 'params:table_destination_path'],
            outputs='message',
            func=organize_images_and_tables,
        ),
        node(
            name='concatenate_tabular_files_node',
            inputs=['params:table_destination_path'],
            outputs='data_filtered',
            func=concatenate_tabular_data,
        )

    ])
