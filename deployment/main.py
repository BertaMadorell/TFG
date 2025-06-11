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
    """
    Forward hook function to capture and store the output (features) of a specific layer during the forward pass.

    Input:
        module: The layer/module to which the hook is attached.
        input: Input to the layer.
        output: Output from the layer.
    """
    global features
    features = output

def backward_hook(module, grad_in, grad_out):
    """
    Backward hook function to capture and store the gradients flowing out of a specific layer during backpropagation.

    Input:
        module: The layer/module to which the hook is attached.
        grad_in: Incoming gradients to the layer.
        grad_out: Outgoing gradients from the layer.
    """
    global gradients
    gradients = grad_out[0]


# Image preprocessing function
def preprocess_image(image_path):
    """
    Preprocesses an input image for model inference.
    Input:
        image_path: Path to the input image file.

    Output: Preprocessed image tensor with a batch dimension, moved to the target device.
    """
    preprocess = transformers['test_transforms']
    image = Image.open(image_path).convert('RGB')  # Open image and convert to RGB
    image = preprocess(image)  # Apply transformations, output shape: (C, H, W)
    return image.unsqueeze(0).to(device)  # Add batch dimension and move to device (CPU/GPU)

# Function to compute the CAM
def compute_cam(input_image, csv_path, f, model_ft):

    """
    Computes the Class Activation Map (CAM) for a given input image using a model.

    Input:
        input_image: Preprocessed image tensor with batch dimension.
        csv_path: Path to the CSV file for looking up ground truth predictions.
        f: Filename corresponding to the input image.
        model_ft: The trained model used for prediction and gradient extraction.

    Output: Normalized CAM heatmap resized to (224, 224).
    """

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
def overlay_cam_on_image(image_path, cam, alpha=0.5):
    """
    Overlays the Class Activation Map (CAM) heatmap on the original image.

    Args:
        image_path: Path to the original image file.
        cam: Normalized CAM heatmap with values between 0 and 1.
        alpha: Transparency factor for the heatmap overlay. Defaults to 0.5.

    Returns: Image with CAM heatmap overlay, RGB format, size (224, 224).
    """
    # Load original image and convert color space from BGR to RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  # Resize to match CAM size

    # Convert CAM from [0,1] float to 8-bit grayscale [0,255]
    cam = np.uint8(255 * cam)

    # Apply a heatmap color map (JET) to the CAM
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency

    # Combine original image and heatmap using weighted sum
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return overlay


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def occlusion_sensitivity(model, image, label_idx, patch_size=7, stride=7):
    """
    Computes an occlusion sensitivity heatmap for a given image and model.
    It systematically occludes patches of the image and measures the drop in the predicted
    probability for the target class, indicating which regions are most important for the prediction.

    Args:
        model: The trained model.
        image: Input image tensor of shape (1, C, H, W).
        label_idx: Index of the target class.
        patch_size: Size of the occlusion patch. Defaults to 7.
        stride: Stride for sliding the occlusion patch. Defaults to 7.

    Returns: Normalized heatmap indicating sensitivity to occlusion (shape H x W).
    """
    _, _, H, W = image.shape
    heatmap = np.zeros((H, W))

    with torch.no_grad():
        output = model(image)
        max_val, class_idx = output.max(1)

        # Get predicted probability for the target class
        probability = torch.exp(max_val).item()

        class_names = ['no_pneumo', 'pneumo']
        print(f"Predicted class index: {class_names[label_idx]}, probability: {probability:.4f}")

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            occluded = image.clone()
            occluded[:, :, y:y+patch_size, x:x+patch_size] = 0.0  # Zero out the patch

            with torch.no_grad():
                occluded = occluded.to(device)
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
    """
    Generates and processes attributions for an input image using the specified explainability method.
    It calculates attributions, dilates them for better visibility, normalizes, and then draws predicted
    and ground truth bounding boxes, saving relevant info to CSVs.

    Input:
        method: Explainability method (e.g., Integrated Gradients) with an attribute() function.
        image_tensor: Input tensor representing the image.
        image: Original image (e.g., as a NumPy array or PIL image).
        ground_truth_csv: Path to CSV containing ground truth bounding boxes.
        img_path: Path to the image file.
        img_name: Name of the image.
        method_name: String name of the explainability method.
        csv_iou: Path to CSV file for Intersection-over-Union (IoU) logging.
        csv_bb: Path to CSV file for bounding box logging.
    """
    
    attributions_list = []

    attributions = metode.attribute(image_tensor, target=1)
    attributions = attributions.squeeze(0).cpu().detach().numpy()
    attributions_list.append(attributions)

    # DILATING RESULTS SO THEY'RE MORE VISIBLE
    attributions_list_dilated = dilate_attributions(attributions_list)

    # SHOWING THE RESULTS
    show_attributions(attributions_list_dilated, imatge, cmap='coolwarm', alpha_map=0.9, alpha_img=0.2, title=nom_metode+'Dilated')


def getPrediction(filename):
    """
    Loads a trained binary classification model and performs prediction on a single image.
    Additionally, generates and saves interpretability visualizations such as Class Activation Maps (CAM) and occlusion sensitivity maps.

    Input:
        filename : Name of the image file located in the 'static' directory (e.g., 'example.png').

    Output:
        predicted_class : Predicted class label (0 or 1) for the input image.
        probability_gc : Confidence probability for the predicted class based on Grad-CAM.
        probability_occ : Confidence probability based on occlusion sensitivity analysis.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model_ft = Classifier()
    model_ft.load_state_dict(torch.load('/home/bertam/CHALLENGE/deployment/models/model_final.pth', map_location='cuda'))
    model_ft.to(device)
    model_ft.eval()

    img_pth = '/home/bertam/CHALLENGE/deployment/static/' + filename

    # Preprocess the image
    image_tensor = preprocess_image(img_pth)
    image_tensor = image_tensor.to(device)

    # Prepare image for overlay
    imatge = cv2.imread(img_pth)
    imatge = cv2.cvtColor(imatge, cv2.COLOR_BGR2RGB)
    imatge = cv2.resize(imatge, (224, 224))

    # Register hooks for Grad-CAM
    target_layer = model_ft.model.layer4
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # Generate CAM and predict
    cam, predicted_class, probability_gc = compute_cam(image_tensor, model_ft)

    # Overlay CAM on image and save
    overlay_cam_on_image(img_pth, cam, output_path=f'static/images/overlay_{filename}')

    # Compute and save occlusion sensitivity heatmap
    heatmap_occ, probability_occ = occlusion_sensitivity(model_ft, image_tensor, label_idx=1)
    heatmap_resized = cv2.resize(heatmap_occ, (imatge.shape[1], imatge.shape[0]))
    plt.imshow(imatge)
    plt.imshow(heatmap_resized, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title("Occlusion Sensitivity")
    plt.savefig(f'static/images/occlusion_{filename}', bbox_inches='tight', pad_inches=0)
    plt.close()

    return predicted_class, probability_gc, probability_occ

