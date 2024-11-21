"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.19.9
"""

# Move jpegs to separate folder with images
import shutil
import os
import glob
import re

def organize_images_and_tables(
        data_source: str,
        image_dest: str,
        tabular_dest: str) -> None:
    
    """For raining data comes in one batch with mixed original images, contrasted images, 
    and images with hand-draw outlined predicted leaf shapes, as well as tabular data containing
    individual leaves locations, as well as calculated leaf area and predicted leaf area.
    For training only original images need to be kept, and tabular data are being moved to another folder,
    From wchich they can be later conactenated into one training csv file.

    Arguments:
        data_source (string): path to folder containing mixed tabular and image data.
        image_dest (string): path to folder in which to store image data.
        data_source (string): path to folder in which to strore tabular data.
    """
    
    os.makedirs(image_dest, exist_ok=True)
    os.makedirs(tabular_dest, exist_ok=True)

    jpeg_files = glob.glob(os.path.join(data_source, '*.JPG'))
    
    # Files are xls but in reality these are csv files.
    xls_files = glob.glob(os.path.join(data_source, '*.xls'))
    
    # Images with (1) are all transformations of the original image.
    pattern = r"\(\d+\)"
    image_files = [f for f in jpeg_files if not re.search(pattern, f)]

    for f in image_files:
        shutil.copy(f, image_dest)
    for f in xls_files:
        shutil.copy(f, tabular_dest)
    message = "File sorting complete."

    return message



# def concatenate_tabular_data():
#     """
#     [TODO]
#     """
#     df = pd.DataFrame()
#     for f in xls_files_staged:
#         try:
#             data = pd.read_csv(f, sep='\t')
#         except Exception as e:
#             print("Data did not read properly")
#         try:
#             label, leaf_est_area, leaf_area = calculate_total_area(data)
#         except Exception as e:
#             print(e)
#             print(f"Something went wrong in {label}")
#         df_row = pd.DataFrame(
#             {
#                 'label': label,
#                 'est_area': [leaf_est_area],
#                 'area': [leaf_area],
#             }
#         )
#         df= pd.concat([df, df_row], ignore_index=True)



# def create_image_path_column():
#     """
#     Create a column in the dataset with reference path to images.
#     """
#     df['label'] = [f.replace('.P', '') for f in df['label']]
#     df['image_path'] = [f"{image_dest}/{label}" for label in df['label']]
#     df['image_path'] = [f.replace('.jpg', '.JPG') for f in df['image_path']]
#     df['area_lost'] = df['est_area'] - df['area']
#     df.to_csv(os.path.join('../data/processed','data.csv'))



# def filter_dataset():
#     image_dest = '../data/staged/images'
#     image_list = glob.glob(os.path.join(image_dest, "*"))
#     data['img_id_exists'] = [f in image_list for f in data['image_path']]
#     data_filtered = data[data['img_id_exists'] == True]
#     data_filtered.to_csv(os.path.join('../data/processed','data_filtered.csv'))



# def check_consistency():
#     img_path_check = [is_image_from_path_readable(image_path=f) for f in data_filtered['image_path']]
#     sum(img_path_check) == len(img_path_check)