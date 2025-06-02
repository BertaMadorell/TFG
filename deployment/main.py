import numpy as np
from PIL import Image
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms 
from torchvision import models
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),  # Redueix les característiques a 512
            nn.ReLU(),  # Activa les neurones no linealment
            nn.Dropout(0.4),  # Redueix overfitting eliminant connexions en cada forward pass
            nn.Linear(512, 2),  # Redueix a 2 classes (Pneumonia o No Pneumonia)
            nn.LogSoftmax(dim=1)  # Converteix les sortides en probabilitats logarítmiques
        )

        self.model.fc = self.classifier    
    
    def forward(self, x):
        return self.model(x)

noms_classes= ['NO PNEUMONIA', 'PNEUMONIA']

transformers = {
'test_transforms' : transforms.Compose([
    transforms.Resize(256),          # Redimensiona la part curta a 256
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])}
# GRAD CAM A MÀ

# Hooks to capture features and gradients
features = None
gradients = None

def forward_hook(module, input, output):
    global features
    features = output

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

# Image preprocessing function
def preprocess_image(image_path):
    preprocess = transformers['test_transforms']
    image = Image.open(image_path).convert('RGB')  # Obrir imatge i convertir-la a RGB
    image = preprocess(image)  # Tensor 3D: (C, H, W)
    return image.unsqueeze(0).to(device)  # Afegir dimensió batch

# Function to compute the CAM
def compute_cam(input_image, model_ft):

    # Forward pass to get predictions
    output = model_ft(input_image)
    max_val, class_idx = output.max(1)

    # Automatically select the class with the highest score
    #probability = max_val.exp().item()
    probability = torch.exp(max_val).item()
    
    predicted_class = noms_classes[class_idx]
    
    # Backward pass for the predicted class
    model_ft.zero_grad()
    output[0, class_idx].backward(retain_graph=True)

    if gradients is None:
        raise RuntimeError("Gradients are not being captured. Check hooks and model layers.")

    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * features, dim=1, keepdim=True)

    # Apply ReLU and normalize the CAM
    cam = F.relu(cam).squeeze().cpu().detach().numpy()
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / cam.max()
    return cam, predicted_class, probability

# Function to overlay the CAM on the original image
def overlay_cam_on_image(image_path, cam, output_path='static/images/overlay.png', alpha=0.5):
    # Load original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    # Normalize and convert CAM
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay heatmap
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    # Save overlay image
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, overlay_bgr)

    return output_path  # Return path to saved image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def occlusion_sensitivity(model, image, label_idx, patch_size=7, stride=7):
    _,_, H, W = image.shape
    heatmap = np.zeros((H, W))
    
    with torch.no_grad():
        output = model(image)
        max_val, class_idx = output.max(1)

        # Automatically select the class with the highest score
        #probability = max_val.exp().item()
        probability = torch.exp(max_val).item()
    

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            occluded = image.clone()
            occluded[:, :, y:y+patch_size, x:x+patch_size] = 0.0  # Zero out the patch

            with torch.no_grad():
                occluded= occluded.to(device)
                output2 = model(occluded)
                probs = torch.exp(output2)
                score = probs[0, label_idx].item()

            drop = probability - score
            heatmap[y:y+patch_size, x:x+patch_size] += drop

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    print(f"Max heatmap:{heatmap.max()}")
    heatmap /= heatmap.max()
    return heatmap, probability

def dilate_attributions(attributions_list, kernel_size=9, iterations=1):
    """
    Dilates the attribution maps according obtained from an explainability method.
    Params:
        attributions_list: list with the attribution maps obtained from an explainability method.
        kernel_size: the size of the square kernel used for dilating the attribution maps.
        iterations: number of iterations for the dilation.
    """
    attributions_list_dilated = []
    for i in range(len(attributions_list)):
        attributions_np = attributions_list[i]
        attributions_np = attributions_np.transpose(1,2,0)
        attributions_dilated = cv2.dilate(attributions_np, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
        attributions_dilated = attributions_dilated.transpose(2,0,1)
        attributions_list_dilated.append(attributions_dilated)
    return attributions_list_dilated

def show_attributions(attributions_list, original_imgs, cmap='coolwarm', alpha_map=0.8, alpha_img=0.4, title=None):
    """
    Plots 10 images of the attribution maps overlayed on the original images.
    Params:
        attributions_list: list with the attribution maps obtained from an explainability method.
        original_imgs: list containing the arrays of the original images
        labels: array with the label for each image.
        predictions: array with the prediction for reach image.
        cmap: color map to use. Default 'coolwarm'.
        alpha_map: alpha value for the attribution maps.
        alpha_img: alpha value for the original images.
        title: title for the figure. Default None.
    """
    attr = attributions_list[0]
    if attr.shape[0] == 3:  # channel-first
        attr = np.transpose(attr, (1, 2, 0))
    
    if original_imgs.shape[0] == 3:  # also transpose if needed
        original_imgs = np.transpose(original_imgs, (1, 2, 0))
        
    if attr.ndim == 3 and attr.shape[2] == 3:
        attr = np.mean(attr, axis=2)



    #attributions_channels_merged = np.maximum(attributions_list[i][0], attributions_list[i][1], attributions_list[i][2])
    plt.imshow(attr, cmap=cmap, alpha=alpha_map, interpolation='nearest')
    plt.imshow(original_imgs, cmap='gray', alpha=alpha_img)
    plt.axis('off')
    plt.title(title)
    plt.savefig(f'static/images/{title}', bbox_inches='tight', pad_inches=0)
    plt.close()
    


def explicabilitat(metode, image_tensor, imatge, nom_metode):
    
    attributions_list = []

    attributions = metode.attribute(image_tensor, target=1)
    attributions = attributions.squeeze(0).cpu().detach().numpy()
    attributions_list.append(attributions)

    # DILATING RESULTS SO THEY'RE MORE VISIBLE
    attributions_list_dilated = dilate_attributions(attributions_list)

    # SHOWING THE RESULTS
    show_attributions(attributions_list_dilated, imatge, cmap='coolwarm', alpha_map=0.9, alpha_img=0.2, title=nom_metode+'Dilated')


def getPrediction(filename):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Per a la classificació binària (Thorax - KO_region o LM - VD)
    model_ft = Classifier()
    model_ft.load_state_dict(torch.load('/home/bertam/CHALLENGE/deployment/models/no_fer_Cas_PAAP_def.pth', map_location='cuda'))
    
    model_ft.to(device)
    model_ft.eval()
    
    img_pth = '/home/bertam/CHALLENGE/deployment/static/'+ filename

    ## Preprocessem la imatge d'entrada ##
    image_tensor = preprocess_image(img_pth)
    image_tensor = image_tensor.to(device)

    ## Imatge per mostrar ##
    imatge = cv2.imread(img_pth)
    imatge = cv2.cvtColor(imatge, cv2.COLOR_BGR2RGB)
    imatge = cv2.resize(imatge, (224,224))

    ## Definim la capa que volem veure com classifica, l'última fc o convolucional, mirar: layer4 ##
    target_layer = model_ft.model.layer4
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)


    ## Compute the CAM ##
    cam, predicted_class, probability_gc = compute_cam(image_tensor, model_ft)


    ## Dibuixem els bounding boxes sobre la imatge ##
    overlay_cam_on_image(img_pth, cam, output_path=f'static/images/overlay_{filename}')
    
    heatmap_occ, probability_occ= occlusion_sensitivity(model_ft, image_tensor, label_idx=1)
    heatmap_resized = cv2.resize(heatmap_occ, (imatge.shape[1], imatge.shape[0]))
    plt.imshow(imatge)  # Imagen base
    plt.imshow(heatmap_resized, cmap='jet', alpha=0.5)  # Superposición
    plt.axis('off')
    plt.title("Occlusion Sensitivity")
    plt.savefig(f'static/images/occlusion_{filename}', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    
    return predicted_class, probability_gc, probability_occ
