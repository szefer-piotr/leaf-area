"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.19.9
"""
import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from torchvision import transforms
from torch.utils.data import DataLoader

from model_builder.model_builder import (
    LeafFrameDataset,
    create_sequential,
    replace_final_layer,
    create_model
)

from torchvision import models
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler



def torch_device_test() -> str:
    """
    Sets the available device.
    Returns:
        device (string): string determining th edevice used: 'cuda' or 'cpu'.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] The device used is {device}.')
    
    return device



def load_data(tabular_data_path: str, dataset_name: str) -> pd.DataFrame:
    """Read tabular data from the source folder"""
    data = pd.read_csv(os.path.join(tabular_data_path, dataset_name))
    print(f"Dataset {dataset_name} loded with {data.shape[0]} records.")

    return data



def split_data(
        data: pd.DataFrame, 
        test_size: float = 0.1, 
        val_size: float = 0.15, 
        random_state: int = 42,
        shuffle: bool = True
    ) -> tuple:
    """
    Split the data into train, validation, and test data. It calls sklern's `train_test_split` functions twice to
    create train, validation, and test sets. The initial train test split is set to shuffle.
    Arguments:
        data (pandas dataframe): tabular data with paths to images and target columns.
        test_size: percentage or integer (number of obsesrvations) that will go to train set. Default is 0.10.
        val_size: percentage or integer (number of obsesrvations) that will go to validation set. Default is 0.15
        random state (integer): random seed. Default is 42.
        shuffle (boolean): whether initial train test split is to be shuffled. Default is True.
    Returns:
        tuple: tuple containing train, validation, and test sets 
    """
    X_train_temp, X_test = train_test_split(
        data, test_size=test_size, random_state=random_state, shuffle=shuffle
    )
    data.shape, X_train_temp.shape, X_test.shape
    X_train, X_val = train_test_split(
        X_train_temp, test_size=val_size, random_state=random_state
    )

    print(f"Shape of the whole set: {data.shape}; \nTrain shape: {X_train.shape}; \nValidation shape {X_val.shape}; \nTest set: {X_test.shape}")

    return X_train, X_val, X_test


def create_transformations():
    """
    Creates a ditionary whith three keys: 'train', 'val', and 'test', refering to train, validation and test inmage transformations
    """
    data_transforms = {
        'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    }

    return data_transforms
    


def create_datasets(
        X_train: pd.DataFrame, 
        X_val: pd.DataFrame , 
        X_test: pd.DataFrame, 
        data_transforms: dict,
        target_transformation: callable,
    ) -> tuple:
    """
    Creates a dictonary of image datasets using custom PyTorch dataset LeafFrameDataset.

    Args:
        X_train (pd.DataFrame): training set.
        X_train (pd.DataFrame): training set.
        X_train (pd.DataFrame): training set.
        data_transforms

    """
    image_datasets = {
    'train': LeafFrameDataset(
        X_train,
        transformations=data_transforms['train'],
        target_name='area_lost',
        target_transformation=target_transformation
    ),
    'val': LeafFrameDataset(
        X_val,
        transformations=data_transforms['val'],
        target_name='area_lost',
        target_transformation=target_transformation
    ),
    'test': LeafFrameDataset(
        X_test,
        transformations=data_transforms['test'],
        target_name='area_lost',
        target_transformation=target_transformation
    )
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return image_datasets, dataset_sizes



def create_dataloaders(image_datasets,
                       batch_size: int = 8):
    """Create dataloaders
    Argumenst:
        image_datasets (dictionary): dictionary containing
        batch_size (integer): size of the batch
    Returns:
        dataloaders (dictionary): train and validation dataloaders of DataLoader PyTorch type.
        under the 'train' and 'val' keys.
    """
    dataloaders = {
    'train': DataLoader(
        dataset=image_datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
        ),
    'val': DataLoader(
        dataset=image_datasets['val'],
        batch_size=batch_size,
        num_workers=0
        )    
    }

    return dataloaders



def instantiate_model(models_dict: dict,
                      model_definition_params: dict,
                      final_layer_params: dict,
                      device: str = 'cpu'
    ):
    """
    Initialize model for transfer learning and allows to create a sequential. It allows to define number of layers
    and nodes to build a nn.Sequential model for the last (usually classification layer) of any nn.Module model.

    Args:
        device (torch.device): cuda or cpu
        models_dict (dict[str: nn.Module]): dictionary with keys refering to callable nn.Models
        model_name (str): key for the model
        weights: str,
        num_layers: int, 
        nodes_per_layer: list,
        activation: nn.Module,
        final_activation: nn.Module,
        fine_tune: bool = True

    Returns:
        nn.Module:
    """
    # model = models.resnet18(weights=weights)
    model = create_model(models_dict, **model_definition_params)
    # Newly added layers have grad = True
    model = replace_final_layer(model, **final_layer_params)
    model = model.to(device)
    print(model)
    
    return model



# def train_model():
#     from torcheval.metrics import R2Score
# loss_fn = MSELoss()
# optimizer = Adam(model.parameters(), amsgrad=True)
# lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# metric = R2Score(device=device)



# import time
# from tempfile import TemporaryDirectory
# import os

# def train_model(
#     model,
#     loss_fn,
#     optimizer,
#     scheduler,
#     num_epochs=2
# ):
#     since = time.time()

#     with TemporaryDirectory() as tempdir:
#         best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
#         torch.save(model.state_dict(), best_model_params_path)
#         best_r2 = 0.0
        
#         for epoch in range(num_epochs):
#             print(f'Epoch {epoch}/{num_epochs - 1}')
#             print('-' * 10)

#             for phase in ['train', 'val']:
#                 if phase == 'train':
#                     model.train()
#                 else:
#                     model.eval()

#                 running_loss = 0.0
#                 running_r2 = 0.0

#                 for inputs, targets in dataloaders[phase]:
#                     inputs = inputs.to(device)
#                     targets = targets.to(device)

#                     optimizer.zero_grad()

#                     with torch.set_grad_enabled(phase == 'train'):
#                         outputs = model(inputs)[:, 0]
#                         loss = loss_fn(outputs, targets)
#                         r2 = metric.update(outputs, targets).compute()

#                         if phase == 'train':
#                             loss.backward()
#                             optimizer.step()

#                     running_loss += loss.item() * inputs.size(0)
#                     running_r2 += r2 * inputs.size(0)

#                 if phase == 'train':
#                     scheduler.step()

#                 epoch_loss = running_loss / dataset_sizes[phase]
#                 epoch_r2 = running_r2 / dataset_sizes[phase]

#                 print(f'{phase} Loss (MSE): {epoch_loss:.4f}, R2: {epoch_r2:.4f}')

#                 if phase == 'val' and epoch_r2 > best_r2:
#                     best_r2 = epoch_r2
#                     torch.save(model.stat_dict(), best_model_params_path)
#             print()

#         time_elapsed = time.time() - since
#         print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        
#         model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    
#     return model


# def validate_model():
#     pass