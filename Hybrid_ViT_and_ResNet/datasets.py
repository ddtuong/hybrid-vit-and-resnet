from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
from PIL import Image
import pandas as pd
from glob import glob
from torchvision import transforms
from configurations import IMG_SIZE, LABELS

# image processing
class IdentityTransform:
    def __call__(self, x):
        return x
    
transforms_train = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        
        transforms.RandomChoice([
        transforms.RandomHorizontalFlip(p=1.0),   
        transforms.RandomVerticalFlip(p=1.0),     
        transforms.RandomRotation(30),            
        transforms.RandomResizedCrop(
            IMG_SIZE, scale=(0.8, 1.2), ratio=(0.9, 1.1)
        ),
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.2, 0.2),   # width_shift_range=0.2, height_shift_range=0.2
            scale=(0.8, 1.2)        # zoom_range=0.2  → from 1-0.2=0.8 to 1+0.2=1.2
        ),
        IdentityTransform()
    ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ]
)

transforms_val = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ]
)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms=None):
        self.df_data = df.values # DataFrame
        self.transforms = transforms

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, index):
        img_path, label = self.df_data[index]
        img = Image.open(img_path).convert("RGB")
    
        if self.transforms is not None:
            img_trans = self.transforms(img)

        return img_trans, label
        
def data_preprocessing(folder_path):
    df = {
        "img_paths": [],
        "labels": []
    }
    for class_ in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_, "*")
        df["labels"] += [LABELS[class_]] * len(glob(class_path))
        df["img_paths"] += glob(class_path)

    return pd.DataFrame(df).sample(frac=1).reset_index(drop=True)