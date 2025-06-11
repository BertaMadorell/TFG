import os
import re
import copy
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import PIL

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader, sampler, random_split
from scipy.stats import chi2_contingency

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import wandb


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset metadata
data = pd.read_csv('/home/bertam/CHALLENGE/stage2_train_metadata.csv')
print(data.head())

data_p = {'filename': data['patientId'], 'pos': data['position'], 'target': data['Target']}
data_p = pd.DataFrame(data_p)
print(data_p.head())



# Confusion matrix and classification report function
def matriu_de_confusio_i_classificacio_report(all_labels, all_preds, filename):
    """
    Generates and saves a confusion matrix plot and prints a classification report.

    Input:
        all_labels: True class labels.
        all_preds: Predicted class labels.
        filename: Path to save the confusion matrix image (as .png or other supported format).
    """
    conf_mat = confusion_matrix(all_labels, all_preds)
    target_names = ['no_pneumo', 'pneumo']
    
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=target_names)
    disp.plot()
    plt.savefig(filename)  # Save the image
    print(f"Saved confusion matrix to {filename}")
    plt.close()
    
    clas_rep = classification_report(all_labels, all_preds, target_names=target_names)
    print(clas_rep)

# Define paths
path = '/home/bertam/CHALLENGE/resnet19_'
path_correct_pn = os.path.join(path, 'correct_class/pneumo')
path_correct_no_pn = os.path.join(path, 'correct_class/no_pneumo')
path_im_pn = os.path.join(path, 'classified_images/pneumo')
path_im_no_pn = os.path.join(path, 'classified_images/no_pneumo')


# Load classified image filenames
im_pneumonia = os.listdir(path_im_pn)
im_no_pneumonia = os.listdir(path_im_no_pn)
im_junt = im_pneumonia + im_no_pneumonia

# Remove file extensions (e.g. '.png')
im_tot = [os.path.splitext(filename)[0] for filename in im_junt]
print(im_tot[:10])


# Function to calculate prediction
def calcular_pred(row):
    """
    Calculates the predicted class label based on whether the original classification was correct.

    Input: 
        row : A row from a DataFrame containing at least two columns:
            - 'corr': 1 if the prediction was correct, 0 if incorrect
            - 'target': the true label (0 or 1)

    Output:
        The predicted label: returns 'target' if correct, or the opposite label if incorrect.
    """
    if row['corr'] == 0:
        return 1 - row['target']
    else:
        return row['target']



# Use regex to remove trailing numbers or extensions if needed
im_tot = [re.sub(r'\.png.*$', '', filename) for filename in im_tot]
print(im_tot[:10])
print(len(im_tot))


# Filter dataframe for only files in im_tot list
f_data_p = data_p[data_p['filename'].isin(im_tot)]
print(len(f_data_p))
print(f_data_p.head())


# Load filenames of correctly classified images
corr_pneumonia = os.listdir(path_correct_pn)
corr_no_pneumonia = os.listdir(path_correct_no_pn)

corr_junt = corr_pneumonia + corr_no_pneumonia
print(len(corr_junt))

# Remove extensions
corr = [os.path.splitext(filename)[0] for filename in corr_junt]
print(corr[:10])


# Add column 'corr' indicating correct classification (1 if in corr list else 0)
f_data_p['corr'] = f_data_p['filename'].apply(lambda x: 1 if x in corr else 0)
print(f_data_p.head())


# Calculate predictions based on correction status and target
f_data_p['pred'] = f_data_p.apply(calcular_pred, axis=1)
print(f_data_p.head())


# Prepare subset dataframe with position and correctness
f_data_p_pc = pd.DataFrame({'pos': f_data_p['pos'], 'correct': f_data_p['corr']})
print(f_data_p_pc.head())


# Prepare dataframe for metrics: position, predictions, and labels
data_mtx_p = pd.DataFrame({'pos': f_data_p['pos'], 'all_preds': f_data_p['pred'], 'all_labels': f_data_p['target']})
print(data_mtx_p.head())


# Filter by position (AP and PA)
data_mtx_ap = data_mtx_p[data_mtx_p['pos'] == 'AP']
print(data_mtx_ap.head())

data_mtx_pa = data_mtx_p[data_mtx_p['pos'] == 'PA']
print(data_mtx_pa.head())


# Count how many 1s and 0s for 'correct' per position
counts_p = f_data_p_pc.groupby('pos')['correct'].value_counts().unstack(fill_value=0)
print(counts_p)


# Calculate percentage for correct/incorrect per position
percentages_p = f_data_p_pc.groupby('pos')['correct'].value_counts(normalize=True).unstack(fill_value=0) * 100
print(percentages_p)


# Chi-square contingency table and test
data_2 = {
    'incorrecte': [430, 173],  # Incorrect classification
    'correcte': [925, 1311]    # Correct classification
}
contingency_table_2 = pd.DataFrame(data_2, index=['AP', 'PA'])
print(contingency_table_2)

chi2_2, p_value_2, dof_2, expected_2 = chi2_contingency(contingency_table_2)

print(f'Chi-square: {chi2_2}')
print(f'p-value: {p_value_2}')
print(f'Degrees of Freedom: {dof_2}')
print(f'Expected table:\n{expected_2}')


# Generate confusion matrices and reports
matriu_de_confusio_i_classificacio_report(
    data_mtx_p['all_labels'], data_mtx_p['all_preds'], 'matriu_general_3_bona_bona.png'
)

matriu_de_confusio_i_classificacio_report(
    data_mtx_ap['all_labels'], data_mtx_ap['all_preds'], 'matriu_Ap_3_bona_bona.png'
)

matriu_de_confusio_i_classificacio_report(
    data_mtx_pa['all_labels'], data_mtx_pa['all_preds'], 'matriu_pa_3_bona_bona.png'
)


# Additional confusion matrix outputs (optional)
confusion_matrix(data_mtx_p['all_labels'], data_mtx_p['all_preds'])
confusion_matrix(data_mtx_ap['all_labels'], data_mtx_ap['all_preds'])
confusion_matrix(data_mtx_pa['all_labels'], data_mtx_pa['all_preds'])
