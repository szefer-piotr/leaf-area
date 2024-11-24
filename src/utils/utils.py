import pandas as pd
import importlib

def calculate_total_area(file: pd.DataFrame) -> pd.DataFrame:
    """
    For all leaves calculate total area, and total estimated area of leaf blades. Estimations were made by hand in the ImageJ software, 
    where outline of each leaf blade were made by hand.
    Arguments:
        data (dataframe): raw output from the ImageJ software after 
    Returns:
        output (tuple): (label, total_estimated_area, total_area) from which the lost area can be determined.
    """
    data_label = file['Label'].unique().item()
    
    n_rows = file.shape[0]
    
    if n_rows % 2 != 0:
        print(f'In {data_label} the number of rows not even.')
        return

    area =  file['Area']
    total_estimated_area = area[:int(n_rows/2)].sum()
    total_area = area[int(n_rows/2):].sum()

    return data_label, total_estimated_area, total_area

# from itertools import compress

# def find_file_by_label(
#         label: str, 
#         data_path_list: list = xls_files_staged
#         ) -> pd.DataFrame:
#     """
#     Find given label in file path lists.
#     Arguments:
#         label (string): id of a datapoint.
#         data_path_list (list): list of all data paths to search
#     Returns:
#         data (pd.DataFrame): 
#     """
#     label = label.replace('.P.jpg', '')
#     bool_list = [label in file_path for file_path in data_path_list]
#     res = list(compress(data_path_list, bool_list))
#     data = pd.read_csv(res[0], sep='\t')
#     return data

# data.head()

# def is_image_from_path_readable(image_path):
#     try:
#         img = Image.open(image_path)
#     except Exception as e:
#         print(e)
#         return False
#     return True

def string_to_callable(callable_string):
    """
    Convert a string to a callable. This function helps to convert Kedro's
    parameters into callables that can be passed to model building functions
    in pipeline.
    """
    if callable_string == 'None':
        return None
    module_name, function_name = callable_string.rsplit('.', 1)

    print(f'[INFO] Module called {module_name} with a function name: {function_name}')

    module = importlib.import_module(module_name)
    func = getattr(module, function_name)
    return func



def instantiate_class(class_name, parameters):
    pass