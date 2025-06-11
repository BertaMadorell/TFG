import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def obtenir_bounding_box(nom_im, overlay_image):
    """
    Draws ground truth bounding boxes on an overlay image based on annotations from a CSV file.

    Input:
        nom_im : Filename of the image (with extension) for which to retrieve and draw bounding boxes.
        overlay_image : The image (already resized or preprocessed) on which the bounding boxes will be drawn. If None, the function will print a message and return.
    """
    root_path = '/home/bertam/CHALLENGE/stage_2_train_labels.csv'
    csv_bb = pd.read_csv(root_path)
    
    # Get the image filename without extension
    img_busc = os.path.splitext(nom_im)[0]

    data = [] 
    for idx, row in csv_bb.iterrows():
        if row[0] == img_busc:
            fila_formatada = ",".join([str(val) if pd.notna(val) else "NaN" for val in row])
            data.append(fila_formatada)

    if overlay_image is not None:
        # Get dimensions of the overlay image
        overlay_h, overlay_w, _ = overlay_image.shape

        # Original image dimensions (RSNA dataset)
        original_h, original_w = 1024, 1024

        # Scaling factors to map original coordinates to overlay size
        scale_x = overlay_w / original_w
        scale_y = overlay_h / original_h

        for line in data:
            parts = line.split(',')
            x, y, w, h = map(float, parts[1:5])  # Coordenades
            category = parts[5]  # Categoria
            
            # Handle NaN values by setting to 0
            x = 0 if np.isnan(x) else x
            y = 0 if np.isnan(y) else y
            w = 0 if np.isnan(w) else w
            h = 0 if np.isnan(h) else h
            
            # Rescale coordinates to match overlay image
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)

            # Draw rectangle and label
            cv2.rectangle(overlay_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(overlay_image, f"Lesio {category}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)

        # Save and display the debug image
        debug_path = "debug_overlay_with_boxes.jpg"
        cv2.imwrite(debug_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))

        # Show image with bounding box
        plt.imshow(overlay_image)
        plt.title(nom_im)
        plt.axis('off')
        plt.show()

        # Errase the file after showing it
        if os.path.exists(debug_path):
            os.remove(debug_path)
    else:
        print(f"No s'ha trobat l'overlay per a la imatge: {nom_im}")
