# TFG
Code for my bachelor's thesis: Development of a prototype tool for assisted diagnosis of pneumonia using chest images

Best model -> This folder contains the weights for the final model.

deployment -> Code related to deploying the final trained model.

appa_arreg ->	Folder where the classified, missclassified and correct classified images are stored. As well as the the CSV of the iou for every XAI method, and the bounding boxes found by the different XAI methods.

Best_model/	Contains the best-performing model selected from comparisons.

obtenir_images_lesions_python.py -> Extracts bounding boxes from the annotated lesions in the CSV files.

Comparacio_models.ipynb ->	Compares different model architectures and XAI methods.

Codi_simple.py -> Functions for iou extraction

Explicabilitat_metodes.ipynb ->	Applies and evaluates various explainability (XAI) methods.

mirar_AFECT_VAR.py	-> Analyzes possible data bias and variable influence.

training.py	-> Training script of the final model.

resultats_ap_pa.ipynb	-> Evaluating metrics for the final model.

stage2_train_metadata.csv	-> Metadata for training images.

stage2_test_metadata.csv	-> Metadata for test images.

stage_2_train_labels.csv	-> Labels for training data.

stage_2_detailed_class_info.csv	-> Detailed class information for the dataset.