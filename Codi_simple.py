#!/usr/bin/env python
# coding: utf-8

# # PNEUMONIA DETECTION (2 CLASSES) WEIGHTS AND BIASES EFFICIENTNET B0

import os
import csv
import time
import copy
import cv2
import torch
import wandb
import PIL
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import torchvision

from PIL import Image
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split, sampler
from sklearn.model_selection import train_test_split



def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.
    Input:
        box1: [x, y, w, h]
        box2: [x, y, w, h]
        
    Output: IoU value, intersection coordinates
    """
    # Intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    # Height and width of the intersection
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    # Area of both bounding boxes
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    # Union
    union_area = box1_area + box2_area - inter_area

    # IoU
    if union_area == 0:
        return 0
    return inter_area / union_area, (x1, y1, x2, y2)


def read_csv(csv_path):
    """
    Reads a CSV file and returns a dictionary grouping bounding boxes by image name.
    
    Input:
        csv_path: Path to the CSV file.
        
    Output: [x, y, w, h] of the bounding boxes in the CSV
    """
    bounding_boxes = {}
    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Saltar l'encapçalament
        for row in csv_reader:
            if (len(row)==6):
                image_name, x, y, w, h, _ = row
            else:
                image_name, x, y, w, h = row
            # Comprova que els valors no estiguin buits
            if x and y and w and h:
                x, y, w, h = int(float(x)), int(float(y)), int(float(w)), int(float(h))
                if image_name not in bounding_boxes:
                    bounding_boxes[image_name] = []
                bounding_boxes[image_name].append([x, y, w, h])
                
    return bounding_boxes


def normalitza_bb(bb):
    """
    Normalizes bounding box coordinates from 255-scale to 1024-scale.
    
    Input:
        bb: Bounding box [x, y, w, h] list
        
    Output: List of normalized bounding box
    """
    return [int(coord * float(1024/255)) for coord in bb]


def draw_intersections(image_path, ground_truth_boxes, predicted_boxes, csv_iou):
    """
    Draws intersections between predicted and ground truth bounding boxes on an image.
    
    Input:
        image_path: Path to the image file.
        ground_truth_boxes: List of ground truth boxes [[x, y, w, h], ...]
        predicted_boxes: List of predicted boxes [[x, y, w, h], ...]
        csv_iou: Path to the output CSV for IoU logging.
        
    Output: Image with drawn bounding boxes and intersections of None.
    """
    # Comprova si el fitxer existeix
    if not os.path.exists(image_path):
        print(f"Error: La imatge {image_path} no existeix.")
        return None

    # Carrega la imatge
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No es pot llegir la imatge {image_path}.")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print('IOU>0:')
    # Itera sobre les boxes de veritat i les prediccions
    for gt_box in ground_truth_boxes:
        for pred_box in predicted_boxes:
            pred_box= normalitza_bb(pred_box)
            iou, intersection_coords = calculate_iou(gt_box, pred_box)
            
            with open(csv_iou, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                #csv_writer.writerow([image_name,iou])
            
            cv2.rectangle(image, (pred_box[0], pred_box[1]), (pred_box[0]+pred_box[2], pred_box[1]+pred_box[3]), (255,0,0), 5) 
            
            # Dibuixa la intersecció en blau si hi ha una intersecció
            if iou > 0:
                print(iou)
                # Assegura't que les coordenades siguin enters
                x1, y1, x2, y2 = map(int, intersection_coords)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), -1)
        
        cv2.rectangle(image, (gt_box[0], gt_box[1]), (gt_box[0]+gt_box[2], gt_box[1]+gt_box[3]), (0,255,0), 5)                

    return image
        


