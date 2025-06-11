#!/usr/bin/env python
# coding: utf-8

"""
PNEUMONIA DETECTION (Binary classification: Pneumonia vs. No Pneumonia)
Using Transfer Learning with ResNet50 and Weights & Biases (wandb) for logging.
"""

# -----------------------------
# Imports and Setup
# -----------------------------

import os
import time #To track the running time of our model
import copy
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #To visualize the data
import matplotlib.image as mpimg

from PIL import Image
import PIL # To load image data to Python
import scipy.ndimage as nd

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader, sampler, random_split
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torchvision import models
from tqdm import tqdm

import wandb
warnings.filterwarnings('ignore')

# -----------------------------
# Device Configuration
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
wandb.init(project="PA simple") # Save plots at platform wandb, project "PA"


# -----------------------------
# Load Data and Metadata
# -----------------------------

root_path = '/home/nuria/rsna-pneumonia-challenge/dataset_three_class/'
train_pneumo_images = os.path.join(root_path, 'images/train')
val_pneumo_images = os.path.join(root_path, 'images/val')
image_data = pd.read_csv('/home/bertam/CHALLENGE/stage2_train_metadata.csv')

print("Class distribution:", image_data['Target'].value_counts())
print(image_data.head())


# 0 is no pneumo and 1 is pneumo
# -----------------------------
# Assign full image paths
# -----------------------------

image_data['full_path'] = ' '
train_rows_pa, val_rows_pa = [], []
train_rows_ap, val_rows_ap = [], []

for idx, row in image_data.iterrows():
    filename = row['patientId'] + '.png'
    # I have to check if the image is in the train or val directory
    train_image_path = os.path.join(train_pneumo_images, filename)
    val_image_path = os.path.join(val_pneumo_images, filename)
    # I check if the path exists
    # I separate the images depending on their position "AP" or "PA"
    if os.path.isfile(train_image_path):
        if row['position']=='PA':
            image_data.loc[idx,'full_path']= train_image_path
            train_rows_pa.append({'path_im': train_image_path, 'label': row['Target']})
        elif row['position']=='AP':
            image_data.loc[idx,'full_path']= train_image_path
            train_rows_ap.append({'path_im': train_image_path, 'label': row['Target']})
    else:
        if row['position']=='PA':
            val_image_path= os.path.join(val_pneumo_images,p)
            image_data.loc[idx,'full_path']= val_image_path
            val_rows_pa.append({'path_im': val_image_path, 'label': row['Target']})
        elif row['position']=='AP':
            val_image_path= os.path.join(val_pneumo_images,p)
            image_data.loc[idx,'full_path']= val_image_path
            val_rows_ap.append({'path_im': val_image_path, 'label': row['Target']})

# Convert to DataFrames
train_data_pa, train_data_ap = pd.DataFrame(train_rows_pa), pd.DataFrame(train_rows_ap)
val_data_pa, val_data_ap = pd.DataFrame(val_rows_pa), pd.DataFrame(val_rows_ap)

# Remove duplicates
for df in [train_data_pa, train_data_ap, val_data_pa, val_data_ap]:
    df.drop_duplicates(inplace=True)

# -----------------------------
# Downsample to balance classes
# -----------------------------

print(train_data_pa['label'].value_counts())
print(val_data_pa['label'].value_counts())
print(train_data_ap['label'].value_counts())
print(val_data_ap['label'].value_counts())

# Downsample AP data
train_0_ap = train_data_ap[train_data_ap['label'] == 0]
train_1_ap = train_data_ap[train_data_ap['label'] == 1]
data_train_balanced_ap = pd.concat([
    resample(train_0_ap, replace=False, n_samples=1084, random_state=42),
    resample(train_1_ap, replace=False, n_samples=1084, random_state=42)
])

# Downsample PA data
train_0_pa = train_data_pa[train_data_pa['label'] == 0]
data_train_balanced_pa = pd.concat([
    resample(train_0_pa, replace=False, n_samples=1084, random_state=42),
    train_data_pa[train_data_pa['label'] == 1]
])

# Downsample validation data
val_0_ap = val_data_ap[val_data_ap['label'] == 0]
val_1_ap = val_data_ap[val_data_ap['label'] == 1]
data_val_balanced_ap = pd.concat([
    resample(val_0_ap, replace=False, n_samples=264, random_state=42),
    resample(val_1_ap, replace=False, n_samples=264, random_state=42)
])

val_0_pa = val_data_pa[val_data_pa['label'] == 0]
data_val_balanced_pa = pd.concat([
    resample(val_0_pa, replace=False, n_samples=264, random_state=42),
    val_data_pa[val_data_pa['label'] == 1]
])

# Final training and validation datasets
train_data = pd.concat([data_train_balanced_ap, data_train_balanced_pa]).sample(frac=1, random_state=42).reset_index(drop=True)
val_data = pd.concat([data_val_balanced_ap, data_val_balanced_pa]).sample(frac=1, random_state=42).reset_index(drop=True)

#Prove that we did it well
print(val_data['label'].value_counts())
print(train_data['label'].value_counts())

# Split validation to get test set
val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=42)

# -----------------------------
# Dataset and Transforms
# -----------------------------

class CustomDataset(Dataset):
    """
    A custom Dataset class that loads images and their corresponding labels from a DataFrame.
    Each row in the DataFrame should contain the image file path and its label.
    Applies image transformations using torchvision transforms.
    """
    def __init__(self, dataframe, transform=None):
        """
        Initializes the dataset.
        Input:
            dataframe: DataFrame containing image paths and labels.
            transform: Optional transformations to apply to each image.
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.
        Input:
            idx: Index of the sample to retrieve.

        Output:
                image_tensor: Transformed image.
                label: Ground truth label (0 or 1).
                filename: Base name of the image file.
        """
        img_path = self.dataframe.iloc[idx, 0]  # Get image file path
        image = Image.open(img_path).convert('RGB')  # Open and convert to RGB
        label = self.dataframe.iloc[idx, 1]  # Get label (0 or 1)
        
        if self.transform:
            image = self.transform(image)  # Apply data transformations
        
        return image, label, os.path.basename(img_path)  # Return image, label, filename

# Data augmentation and preprocessing
transformers = {
    'train_transforms': transforms.Compose([
        transforms.Resize(256),  # Redimension the short part of the image to 256
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val_transforms' : transforms.Compose([
        transforms.Resize(256),         
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test_transforms' : transforms.Compose([
        transforms.Resize(256),          
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Apply transforms to datasets
train_trsf = CustomDataset(train_data, transformers['train_transforms'])
val_trsf = CustomDataset(val_data, transformers['val_transforms'])
test_trsf = CustomDataset(test_data, transformers['test_transforms'])

# DataLoader setup
categories = ['train', 'val', 'test']
dset = {'train': train_trsf, 'val': val_trsf, 'test': test_trsf}
dataloaders = {x: DataLoader(dset[x], batch_size=32, shuffle=True, num_workers=0) for x in categories}

# -----------------------------
# Model Definition
# -----------------------------

class Model(nn.Module):
    """
    A custom neural network model based on a pretrained ResNet50 for binary classification (Pneumonia detection).

    - Replaces the ResNet50 classifier with a custom fully connected head.
    - Freezes early layers to focus training on higher-level features and make fine tunning.
    - Includes methods for forward propagation and model training with early stopping and learning rate scheduling.
    - Training metrics and performance are logged using Weights & Biases (wandb).
    """
    def __init__(self):
        """
        Initializes the model by:
        - Loading a pretrained ResNet50.
        - Replacing the default classifier with a custom sequential classifier suited for binary classification.
        - Freezing all layers except 'layer4' and the classifier to fine-tune only the last layers.
        """
        # Load pretrained ResNet50
        super(Model, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        # Replace the default classifier with a custom one
        self.classifier = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),  # Reduces characteristics to 512
            nn.ReLU(),  # Activates the neurons in a non linear way
            nn.Dropout(0.4),  # Reduces overfitting by eliminating connections in each forward pass
            nn.Linear(512, 2)  # Reduces to 2 classes (Pneumonia or No Pneumonia)
            #nn.LogSoftmax(dim=1)  # Converts the outputs into logarithmic probabilities
        )

        # Freeze early layers except 'layer4' and the classifier
        for name, param in self.model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

        # Replace the original fully connected layer
        self.model.fc = self.classifier

    def forward(self, x):
        """
        Defines the forward pass of the model.
        Input: tensor (image batch).
        Output: logits for classification.
        """
        return self.model(x)   
    
    def fit(self, dataloaders, num_epochs):
        """
        Trains the model using a given number of epochs with early stopping and learning rate scheduling.

        - Uses separate learning rates for the last ResNet block (layer 4) and the classifier.
        - Applies gradient clipping to stabilize training.
        - Logs training and validation metrics (loss and accuracy) to wandb.
        - Saves the best model weights based on validation accuracy.

        Input: 
            dataloaders: Containing 'train' and 'val' DataLoaders.
            num_epochs: Number of epochs to train the model.

        Returns: The trained model with the best validation performance.
        """
        train_on_gpu = torch.cuda.is_available()
        # Optimizer with separate learning rates for the final layers
        optimizer = optim.Adam(
            [
                {"params": self.model.layer4.parameters(), "lr": 1e-4},  # Smaller learning rate
                {"params": self.model.fc.parameters(), "lr": 1e-3}       # Bigger learning rate
            ]
        )
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)  # Based in val_acc
        # Loss function
        criterion = nn.CrossEntropyLoss()  # Better than NLLLoss
        
        since = time.time()  
        patience = 10
        counter = 0      
        
        # Save the best model weights
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        if train_on_gpu:
            self.model = self.model.to(device)
        for epoch in range(1, num_epochs + 1):
            print("epoch {}/{}".format(epoch, num_epochs))
            print("-" * 10)            
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()                
                    
                running_loss = 0.0
                running_corrects = 0.0                
                
                for inputs, labels, _ in dataloaders[phase]:
                    if train_on_gpu:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                    optimizer.zero_grad()                    
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)                        
                        
                        if phase == 'train':
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            optimizer.step() 
                            
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)                
                    
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)                
                
                # We send the metrics to Weights & Biases
                if phase == 'train':
                    wandb.log({"Train Loss": epoch_loss, "Train Accuracy": epoch_acc, "epoch": epoch})
                else:
                    wandb.log({"Validation Loss": epoch_loss, "Validation Accuracy": epoch_acc, "epoch": epoch})                
                    
                print("{} loss:  {:.4f}  acc: {:.4f}".format(phase, epoch_loss, epoch_acc))               
                
                # Early stopping based on validation accuracy
                if phase == 'val':
                    scheduler.step(epoch_loss)
                
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(self.model.state_dict())        
                        counter = 0
                    else:
                        counter += 1
                    
                    if counter >= patience:
                        print("🛑 Early stopping activat!")
                        self.model.load_state_dict(best_model_wts)  # Get the best model
                        return self.model
                    
        time_elapsed = time.time() - since
        print('time completed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print("best val acc: {:.4f}".format(best_acc))        
        
        self.model.load_state_dict(best_model_wts)
        return self.model


# -----------------------------
# Train the Model
# -----------------------------

model = Model()
model_trained = model.fit(dataloaders, 300)

# -----------------------------
# Save Final Weights
# -----------------------------

torch.save(model.state_dict(), "./Best_weights/model_final.pth")
print("Model training complete and weights saved.")