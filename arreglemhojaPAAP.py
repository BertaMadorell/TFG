#!/usr/bin/env python
# coding: utf-8

# # PNEUMONIA DETECTION (2 CLASSES) WEIGHTS AND BIASES

# In[1]:


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt #To visualize the data
import copy
import time #To track the running time of our model
import PIL # To load image data to Python
import pandas as pd
import shutil
import os
from torchvision import models
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, sampler, random_split
from PIL import Image
import scipy.ndimage as nd
import wandb

from tqdm import tqdm
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

wandb.init(project="PA simple")


# In[3]:


root_path = '/home/nuria/rsna-pneumonia-challenge/dataset_three_class/'
train_pneumo_images = root_path + 'images/train'
val_pneumo_images = root_path + 'images/val'
image_data = pd.read_csv('/home/bertam/CHALLENGE/stage2_train_metadata.csv')


# In[4]:

print(len(image_data[image_data['Target'] == 0]), len(image_data[image_data['Target'] == 1]))


# Primer miraré de classificar només entre pneumonia i no pneumonia.

# 0 és no pneumo i 1 és sí pneumo

# In[5]:


print(image_data.head())


# Creo una columna nova amb els paths a les imatges, depèn de si la imatge està a la carpeta de train o de val.

# In[6]:


image_data['full_path'] = ' '
train_rows_pa = []
val_rows_pa = []
train_rows_ap = []
val_rows_ap = []
for idx, row in image_data.iterrows():
    p = row['patientId']+'.png'
    
    #Per mirar si la imatge és dins el path de train, escric tot el directori de train + el nom de la imatge
    train_image_path = os.path.join(train_pneumo_images, p)
    # Comprovo si exiteix aquesta ruta, és a dir, miro si la imatge és al directori de train
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
            
train_data_pa= pd.DataFrame(train_rows_pa)
train_data_ap= pd.DataFrame(train_rows_ap)
val_data_pa= pd.DataFrame(val_rows_pa)
val_data_ap= pd.DataFrame(val_rows_ap)
print(train_data_ap.head())
print(val_data_ap.head())
print(train_data_pa.head())
print(val_data_pa.head())


# In[7]:
# In[8]:
train_data_pa=train_data_pa.drop_duplicates()
train_data_ap=train_data_ap.drop_duplicates()
val_data_pa=val_data_pa.drop_duplicates()
val_data_ap=val_data_ap.drop_duplicates()

print(train_data_pa['label'].value_counts())
print(val_data_pa['label'].value_counts())
print(train_data_ap['label'].value_counts())
print(val_data_ap['label'].value_counts())
# Farem un ground truth per mirar si hem classificat bé les imatges o no.

# In[9]:
train_0_ap= train_data_ap[train_data_ap['label'] == 0]
train_0_pa= train_data_pa[train_data_pa['label'] == 0]
val_0_ap= val_data_ap[val_data_ap['label'] == 0]
val_0_pa= val_data_pa[val_data_pa['label'] == 0]

train_1_ap= train_data_ap[train_data_ap['label'] == 1]
val_1_ap= val_data_ap[val_data_ap['label'] == 1]

# In[9]:
from sklearn.utils import resample
# Submuestreo de data_train_pa_0
data_train_0_downsampled_ap = resample(
    train_0_ap,
    replace=False,       # no hacemos muestreo con reemplazo
    n_samples=1084,      # igual al número de clase minoritaria
    random_state=42      # para reproducibilidad
)

# Submuestreo de data_train_pa_0
data_train_1_downsampled_ap = resample(
    train_1_ap,
    replace=False,       # no hacemos muestreo con reemplazo
    n_samples=1084,      # igual al número de clase minoritaria
    random_state=42      # para reproducibilidad
)

# Combinar con la clase minoritaria (que ya está balanceada)
data_train_balanced_ap = pd.concat([data_train_0_downsampled_ap, data_train_1_downsampled_ap])
# In[9]:
# Submuestreo de data_train_pa_0
data_train_0_downsampled_pa = resample(
    train_0_pa,
    replace=False,       # no hacemos muestreo con reemplazo
    n_samples=1084,      # igual al número de clase minoritaria
    random_state=42      # para reproducibilidad
)

# Combinar con la clase minoritaria (que ya está balanceada)
data_train_balanced_pa = pd.concat([data_train_0_downsampled_pa, train_data_pa[train_data_pa['label'] == 1]])
# In[9]:
# Submuestreo de data_train_pa_0
data_val_0_downsampled_ap = resample(
    val_0_ap,
    replace=False,       # no hacemos muestreo con reemplazo
    n_samples=264,      # igual al número de clase minoritaria
    random_state=42      # para reproducibilidad
)
data_val_1_downsampled_ap = resample(
    val_1_ap,
    replace=False,       # no hacemos muestreo con reemplazo
    n_samples=264,      # igual al número de clase minoritaria
    random_state=42      # para reproducibilidad
)
# Combinar con la clase minoritaria (que ya está balanceada)
data_val_balanced_ap = pd.concat([data_val_0_downsampled_ap, data_val_1_downsampled_ap])
# In[9]:
# Submuestreo de data_train_pa_0
data_val_0_downsampled_pa = resample(
    val_0_pa,
    replace=False,       # no hacemos muestreo con reemplazo
    n_samples=264,      # igual al número de clase minoritaria
    random_state=42      # para reproducibilidad
)

# Combinar con la clase minoritaria (que ya está balanceada)
data_val_balanced_pa = pd.concat([data_val_0_downsampled_pa, val_data_pa[val_data_pa['label'] == 1]])


data_train_balanced = pd.concat([data_train_balanced_ap, data_train_balanced_pa])
data_val_balanced = pd.concat([data_val_balanced_ap, data_val_balanced_pa])
# Barajamos el resultado para evitar orden por clase
train_data = data_train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
# Barajamos el resultado para evitar orden por clase
val_data = data_val_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(val_data['label'].value_counts())
print(train_data['label'].value_counts())

gth_0= "./ground_truth/no_pneumo"
gth_1 = "./ground_truth/si_pneumo"


# In[128]:

# Ara tenim el ground truth i els conjunts de train i val, del conjunt de val en treuré un conjunt de test. Utilitzo el 50% de val per test i així tenen la mateixa proporció val i test.

# In[10]:


val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=42)


print(len(train_data[train_data['label'] == 0]), len(train_data[train_data['label'] == 1]))

# Ara farem data augmentation perquè el model generalitzi millor

# In[11]:


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label, os.path.basename(img_path)


# In[12]:


transformers = {'train_transforms' : transforms.Compose([
    transforms.Resize(256),  # Redimensiona la part curta a 256
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]),
'val_transforms' : transforms.Compose([
    transforms.Resize(256),          # Redimensiona la part curta a 256
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]),
'test_transforms' : transforms.Compose([
    transforms.Resize(256),          # Redimensiona la part curta a 256
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])}


# In[13]:


train_trsf = CustomDataset(train_data, transformers['train_transforms'])
val_trsf = CustomDataset(val_data, transformers['val_transforms'])
test_trsf = CustomDataset(test_data, transformers['test_transforms'])


# In[14]:


# Defineix les categories
categories = ['train', 'val', 'test']

# Defineix el diccionari amb els conjunts de dades
dset = {
    'train': train_trsf,  # Conjunt de dades d'entrenament
    'val': val_trsf,      # Conjunt de dades de validació
    'test': test_trsf     # Conjunt de dades de test
}
# Nombre de threads (treballadors)
num_threads = 0  # Canvia el valor si necessites utilitzar més threads

# Defineix el diccionari de dataloaders utilitzant un comprehension
dataloaders = {
    x: DataLoader(dset[x], batch_size=32, shuffle=True, num_workers=num_threads)
    for x in categories
}


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),  # Redueix les característiques a 512
            nn.ReLU(),  # Activa les neurones no linealment
            nn.Dropout(0.4),  # Redueix overfitting eliminant connexions en cada forward pass
            nn.Linear(512, 2)  # Redueix a 2 classes (Pneumonia o No Pneumonia)
            #nn.LogSoftmax(dim=1)  # Converteix les sortides en probabilitats logarítmiques
        )

        # Congela totes les capes excepte layer4 i fc
        for name, param in self.model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        self.model.fc = self.classifier    
    
    def forward(self, x):
        return self.model(x)    
    
    def fit(self, dataloaders, num_epochs):
        train_on_gpu = torch.cuda.is_available()
        optimizer = optim.Adam(
            [
                {"params": self.model.layer4.parameters(), "lr": 1e-4},  # Learning rate petita
                {"params": self.model.fc.parameters(), "lr": 1e-3}       # Learning rate més gran
            ]
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)  # Basat en val_acc
        # Calcula els pesos (ex: [classe_0, classe_1])
        criterion = nn.CrossEntropyLoss()  # Millor que NLLLoss
        since = time.time()  
        
        patience = 10
        counter = 0      
        
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
                
                # Enviem les mètriques específiques a Weights & Biases
                if phase == 'train':
                    wandb.log({"Train Loss": epoch_loss, "Train Accuracy": epoch_acc, "epoch": epoch})
                else:
                    wandb.log({"Validation Loss": epoch_loss, "Validation Accuracy": epoch_acc, "epoch": epoch})                
                    
                print("{} loss:  {:.4f}  acc: {:.4f}".format(phase, epoch_loss, epoch_acc))               
                
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
                        self.model.load_state_dict(best_model_wts)  # Recupera el millor model
                        return self.model
                    
        time_elapsed = time.time() - since
        print('time completed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print("best val acc: {:.4f}".format(best_acc))        
        
        self.model.load_state_dict(best_model_wts)
        return self.model


# Calling the model and fit on training data:
model = Model()
model_ft = model.fit(dataloaders,300)


# Per recarregar els pesos al model quan el vulgui tornar a utilitzar, primer el guardo.

# In[ ]:


torch.save(model.state_dict(), "./Best_weights/no_fer_Cas_PAAP_def.pth")


# Aquest codi serveix per carregar els pesos al model:

print('fi')