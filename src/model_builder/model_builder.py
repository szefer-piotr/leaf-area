from torch.utils.data import Dataset
from PIL import Image
import torch
import torch.nn as nn

class LeafFrameDataset(Dataset):
    def __init__(self, data, transformations, target_name = 'area_lost', 
                 ):
        """
        Arguments:
            dataset (string): Path to the csv file with tabular data.
            transformations (list): List of transformations on the images
            target_name (string): Name of the target column.
        """
        self.data = data
        self.target_name = target_name
        self.transformations = transformations

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data.iloc[idx]['image_path']
        # img = cv2.imread(img_path)
        img = Image.open(img_path)
        
        if self.transformations is not None:
            img = self.transformations(img)
        
        target = self.data.iloc[idx][self.target_name].astype('float32')
        
        return img, target



def create_sequential(num_layers, nodes_per_layer, activation=nn.ReLU, final_activation = None):
    """
    Create a Sequential module with the specified number of layers and nodes.

    Args:
        num_layers (int): Number of layers in the Sequential module.
        nodes_per_layer (list[int]): List of integers specifying the number of nodes in each layer.
                                     Must have `num_layers + 1` elements (input size + output sizes).
        activation (nn.Module): The activation function to use between layers. Default is nn.ReLU.
        final_activation (nn.Module): Optional final activation function after the last layer.

    Returns:
        nn.Sequential: The constructed Sequential module.
    """
    if len(nodes_per_layer) != num_layers + 1:
        raise ValueError('Number of nodes per layer must have the same length as the number of layers + 1!')
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(nodes_per_layer[i], nodes_per_layer[i+1]))
        if i < num_layers-1:
            layers.append(activation())
    if final_activation is not None:
        layers.append(final_activation())
    
    return nn.Sequential(*layers)



def replace_final_layer(
        model, 
        num_layers=4, 
        nodes_per_layer=[64,128,64,32,1],
        activation=nn.ReLU,
        final_activation=None,
    ):
    """
    Replace the final layer of a PyTorch model with a new layer.

    Args:
        model (nn.Module): A PyTorch model instance.
        new_layer (nn.Module): The new layer to replace the final layer with.
    
    Returns:
        nn.Module: The model with the final layer replaced.
    """
    children = list(model.named_children())
    last_name, last_module = children[-1]
    new_layer = create_sequential(
        num_layers, 
        nodes_per_layer,
        activation,
        final_activation)
    
    setattr(model, last_name, replace_final_layer(last_module, new_layer))
    
    return model