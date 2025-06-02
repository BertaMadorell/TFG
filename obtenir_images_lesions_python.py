import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def obtenir_bounding_box(nom_im, overlay_image):
    root_path = '/home/bertam/CHALLENGE/stage_2_train_labels.csv'
    csv_bb = pd.read_csv(root_path)
    
    # Extreu el nom del fitxer sense extensió
    img_busc = os.path.splitext(nom_im)[0]

    data = []  # Inicialitzem una llista per emmagatzemar la informació
    for idx, row in csv_bb.iterrows():
        if row[0] == img_busc:
            fila_formatada = ",".join([str(val) if pd.notna(val) else "NaN" for val in row])
            data.append(fila_formatada)

    if overlay_image is not None:
        # Dimensions de l'overlay
        overlay_h, overlay_w, _ = overlay_image.shape

        # Dimensions originals (modifica segons el teu dataset)
        original_h, original_w = 1024, 1024

        # Escales per ajustar les coordenades
        scale_x = overlay_w / original_w
        scale_y = overlay_h / original_h

        for line in data:
            parts = line.split(',')
            x, y, w, h = map(float, parts[1:5])  # Coordenades
            category = parts[5]  # Categoria
            
            # Substitueix NaN per 0
            x = 0 if np.isnan(x) else x
            y = 0 if np.isnan(y) else y
            w = 0 if np.isnan(w) else w
            h = 0 if np.isnan(h) else h
            
            # Ajusta les coordenades a l'overlay
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)

            # Dibuixa la bounding box sobre l'overlay
            cv2.rectangle(overlay_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(overlay_image, f"Lesio {category}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)

        # Guarda l'overlay amb bounding boxes per depurar
        debug_path = "debug_overlay_with_boxes.jpg"
        cv2.imwrite(debug_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))

        # Mostra la imatge amb bounding boxes
        plt.imshow(overlay_image)
        plt.title(nom_im)
        plt.axis('off')
        plt.show()

        # Esborra el fitxer després de mostrar-lo
        if os.path.exists(debug_path):
            os.remove(debug_path)
    else:
        print(f"No s'ha trobat l'overlay per a la imatge: {nom_im}")
