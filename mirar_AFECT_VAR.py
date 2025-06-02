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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = pd.read_csv('/home/bertam/CHALLENGE/stage2_train_metadata.csv')
print(data.head())
data_p= {'filename': data['patientId'], 'pos': data['position'], 'target': data['Target']}
data_p= pd.DataFrame(data_p)
print(data_p.head())


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report
def matriu_de_confusio_i_classificacio_report (all_labels, all_preds, filename):
    conf_mat = confusion_matrix(all_labels, all_preds)
    target_names = ['no_pneumo', 'pneumo']
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=target_names)
    disp.plot()
    plt.savefig(filename)  # Guarda la imagen
    print(f"Guardada la matriz de confusión en {filename}")
    plt.close()
    clas_rep = classification_report(all_labels, all_preds, target_names=target_names)
    print(clas_rep)


path = '/home/bertam/CHALLENGE/resnet19_'
path_correct_pn= path + '/correct_class/pneumo'
path_correct_no_pn= path + '/correct_class/no_pneumo'
path_im_pn = path + '/classified_images/pneumo'
path_im_no_pn =  path + '/classified_images/no_pneumo'


im_pneumonia = os.listdir(path_im_pn)
im_no_pneumonia = os.listdir(path_im_no_pn)

im_junt = im_pneumonia + im_no_pneumonia

# Eliminar la extensión '.png' (o cualquier otra extensión que desees)
im_tot= [os.path.splitext(filename)[0] for filename in im_junt]
print (im_tot[:10])

import re

# Aplica la condició fila per fila
def calcular_pred(row):
    if row['corr'] == 0:
        return 1-row['target']
    else:
        return row['target']

# Usamos una expresión regular para eliminar la parte "_<número>"
im_tot = [re.sub(r'\.png.*$', '', os.path.splitext(f)[0]) for f in im_tot]


# Verifica los primeros resultados
print(im_tot[:10])
print(len(im_tot))
# Filtra el DataFrame para mantener solo las filas cuyos filenames estén en la lista
f_data_p = data_p[data_p['filename'].isin(im_tot)]

print(len(f_data_p))

# Verifica los primeros registros del nuevo DataFrame filtrado
print(f_data_p.head())
corr_pneumonia = os.listdir(path_correct_pn)
corr_no_pneumonia = os.listdir(path_correct_no_pn)

corr_junt = corr_pneumonia + corr_no_pneumonia
print(len(corr_junt))
# Eliminar la extensión '.png' (o cualquier otra extensión que desees)
corr= [os.path.splitext(filename)[0] for filename in corr_junt]
print (corr[:10])
f_data_p['corr'] = f_data_p['filename'].apply(lambda x: 1 if x in corr else 0)
print(f_data_p.head())

f_data_p['pred'] = f_data_p.apply(calcular_pred, axis=1)
print(f_data_p.head())

f_data_p_pc= {'pos': f_data_p['pos'], 'correct': f_data_p['corr']}
f_data_p_pc= pd.DataFrame(f_data_p_pc)
print(f_data_p_pc.head())

data_mtx_p ={'pos':f_data_p['pos'],'all_preds': f_data_p['pred'], 'all_labels': f_data_p['target']}
data_mtx_p= pd.DataFrame(data_mtx_p)
print(data_mtx_p.head())

data_mtx_ap=data_mtx_p[data_mtx_p['pos']=='AP']
print(data_mtx_ap.head())

data_mtx_pa=data_mtx_p[data_mtx_p['pos']=='PA']
print(data_mtx_pa.head())

# Contar cuántos 1 y 0 hay en 'pred' para cada sexo
counts_p = f_data_p_pc.groupby('pos')['correct'].value_counts().unstack(fill_value=0)

# Mostrar el resultado
print(counts_p)

# Contar cuántos 1 y 0 hay en 'pred' para cada sexo y calcular el porcentaje
percentages_p = f_data_p_pc.groupby('pos')['correct'].value_counts(normalize=True).unstack(fill_value=0) * 100

# Mostrar el resultado
print(percentages_p)
import pandas as pd
from scipy.stats import chi2_contingency

# Crear la tabla de contingencia con los valores de la clasificación correcta e incorrecta
data_2 = {
    'incorrecte': [430, 173],  # Clasificación incorrecta (pred = 0)
    'correcte': [925, 1311]   # Clasificación correcta (pred = 1)
}

# Crear un DataFrame de la tabla de contingencia
contingency_table_2 = pd.DataFrame(data_2, index=['AP', 'PA'])
print(contingency_table_2)
print ( "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
# Realizar el test de chi-cuadrado
chi2_2, p_value_2, dof_2, expected_2 = chi2_contingency(contingency_table_2)

# Mostrar los resultados
print(f'Chi-cuadrado: {chi2_2}')
print(f'Valor p: {p_value_2}')
print(f'Degrees of Freedom (dof): {dof_2}')
print(f'Tabla esperada: \n{expected_2}')

matriu_de_confusio_i_classificacio_report(data_mtx_p['all_labels'],data_mtx_p['all_preds'], 'matriu_general_3_bona_bona.png')

matriu_de_confusio_i_classificacio_report(data_mtx_ap['all_labels'],data_mtx_ap['all_preds'], 'matriu_Ap_3_bona_bona.png')

matriu_de_confusio_i_classificacio_report(data_mtx_pa['all_labels'],data_mtx_pa['all_preds'], 'matriu_pa_3_bona_bona.png')

confusion_matrix(data_mtx_p['all_labels'],data_mtx_p['all_preds'])

confusion_matrix(data_mtx_ap['all_labels'],data_mtx_ap['all_preds'])

confusion_matrix(data_mtx_pa['all_labels'],data_mtx_pa['all_preds'])