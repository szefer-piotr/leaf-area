"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.19.9
"""

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from torchvision import models
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

def read_data():
    import pandas as pd
    data = pd.read_csv('../data/processed/data_filtered.csv')
    data.shape



def split_data():
    from sklearn.model_selection import train_test_split
    X_train_temp, X_test = train_test_split(
        data, test_size=0.10, random_state=42, shuffle=True
    )
    data.shape, X_train_temp.shape, X_test.shape
    X_train, X_val = train_test_split(
        X_train_temp, test_size=0.15, random_state=42
    )
    print(f"Shape of the whole set: {data.shape}; \nTrain shape: {X_train.shape}; \nValidation shape {X_val.shape}; \nTest set: {X_test.shape}")


def create_transformations():
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
    


def create_datasets():
    image_datasets = {
    'train': LeafFrameDataset(
        X_train,
        transformations=data_transforms['train'],
        target_name='area_lost'
    ),
    'val': LeafFrameDataset(
        X_val,
        transformations=data_transforms['val'],
        target_name='area_lost'
    ),
    'test': LeafFrameDataset(
        X_test,
        transformations=data_transforms['test'],
        target_name='area_lost'
    )
}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    pass



def create_dataloaders():
    dataloaders = {
    'train': DataLoader(
        dataset=image_datasets['train'],
        batch_size=8,
        shuffle=True,
        num_workers=0
        ),
    'val': DataLoader(
        dataset=image_datasets['val'],
        batch_size=8,
        num_workers=0
        )
}
    pass


def initialize_model():
    model = models.resnet18(weights='IMAGENET1K_V1')
    # for param in model.parameters():
    # #     param.requires_grad = False
    # # Newly added layers have grad = True
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 1)
    # model = model.to(device)
    # pass



def train_model():
    from torcheval.metrics import R2Score
loss_fn = MSELoss()
optimizer = Adam(model.parameters(), amsgrad=True)
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
metric = R2Score(device=device)



import time
from tempfile import TemporaryDirectory
import os

def train_model(
    model,
    loss_fn,
    optimizer,
    scheduler,
    num_epochs=2
):
    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)
        best_r2 = 0.0
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_r2 = 0.0

                for inputs, targets in dataloaders[phase]:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)[:, 0]
                        loss = loss_fn(outputs, targets)
                        r2 = metric.update(outputs, targets).compute()

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_r2 += r2 * inputs.size(0)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_r2 = running_r2 / dataset_sizes[phase]

                print(f'{phase} Loss (MSE): {epoch_loss:.4f}, R2: {epoch_r2:.4f}')

                if phase == 'val' and epoch_r2 > best_r2:
                    best_r2 = epoch_r2
                    torch.save(model.stat_dict(), best_model_params_path)
            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    
    return model


def validate_model():
    pass