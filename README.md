# Digital Pathology Cell Analysis & Image Classification Pipeline

An end-to-end Computer Vision and Deep Learning pipeline in Python for segmenting, counting, and classifying microscopic biological cell images using OpenCV, PyTorch, and Grad-CAM explainability maps.

## Tech Stack
* **Language:** Python 3.10
* **Computer Vision:** OpenCV (Gaussian Blur, Morphological Transformations, Watershed Algorithm)
* **Deep Learning:** PyTorch, Torchvision (ResNet18 Transfer Learning)
* **Explainability:** Grad-CAM (Gradient-weighted Class Activation Mapping)

## Core Features
1. **Adaptive Image Preprocessing:** Cleans background noise using Otsu thresholding and Gaussian filtering.
2. **Cell Boundary Isolation:** Isolates touching/overlapping cell nuclei using Distance Transforms and Watershed segmentation.
3. **Multi-Class Classification:** Fine-tunes a pre-trained ResNet18 model to classify blood cell subtypes with high confidence.
4. **Visual Interpretability:** Generates Grad-CAM activation heatmaps highlighting regions driving neural network decisions.
