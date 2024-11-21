import shutil
import os
import glob
import re

from utils.utils import calculate_total_area
import pandas as pd

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

    print(f"[INFO] {message}")

    return message



def concatenate_tabular_data(table_destination_path):
    """
    Concatenate csv files for each photo into one dataframe. During concatenation total (for all leaves) predicted area, 
    and real area are also calcualted using `calculate_total_area` function, that outputs three values: predicted area,
    real area and their difference: area lost.

    Arguments:
        table_destination_path (string): path to the folder with tabular data for individual files.
    """

    xls_files_staged = glob.glob(os.path.join(table_destination_path, "*.xls"))

    df = pd.DataFrame()

    for f in xls_files_staged:
        try:
            data = pd.read_csv(f, sep='\t')
        except Exception as e:
            print("Data did not read properly")
        try:
            label, leaf_est_area, leaf_area = calculate_total_area(data)
        except Exception as e:
            print(e)
            print(f"Something went wrong in {label}")
        df_row = pd.DataFrame(
            {
                'label': label,
                'est_area': [leaf_est_area],
                'area': [leaf_area],
            }
        )
        df= pd.concat([df, df_row], ignore_index=True)

    return df



def create_image_path_column(df: pd.DataFrame,
                             image_dest: str,
                             ) -> pd.DataFrame:
    """
    Create a column in the dataset with reference path to images.
    """
    
    df['label'] = [f.replace('.P', '') for f in df['label']]
    df['image_path'] = [f"{image_dest}/{label}" for label in df['label']]
    df['image_path'] = [f.replace('.jpg', '.JPG') for f in df['image_path']]
    df['area_lost'] = df['est_area'] - df['area']
    
    return df



def filter_dataset(data: pd.DataFrame,
                   image_dest: str) -> pd.DataFrame:
    """
    Filters only records, for which an image in the image foleder exists.
    Arguments:
        data (data frame): concatenated datset with column named 'image_path' identyfying corresponding images.
        image_dest (string): path to the image files.
    Returns:
        data_fileterd (data frame): data with records holding existing corresponding images
    """
    image_list = glob.glob(os.path.join(image_dest, "*"))
    data['img_id_exists'] = [f in image_list for f in data['image_path']]
    data_filtered = data[data['img_id_exists'] == True]
    
    return data_filtered



# def check_consistency():
#     img_path_check = [is_image_from_path_readable(image_path=f) for f in data_filtered['image_path']]
#     sum(img_path_check) == len(img_path_check)