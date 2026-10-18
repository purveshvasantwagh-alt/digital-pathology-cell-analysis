
# Blood Cell Image Classification and Segmentation Prototype

A Python prototype for exploring blood-cell image classification and classical image segmentation. It combines an OpenCV watershed segmentation function with a PyTorch ResNet18 classification pipeline.

## Implemented

- Loads four blood-cell classes from class-named image folders.
- Splits the provided training folder into stratified training and validation subsets.
- Trains an ImageNet-pretrained ResNet18 classification head.
- Supports an experiment that also fine-tunes ResNet18's final block (`layer4`).
- Evaluates a saved checkpoint and writes per-class metrics and a confusion matrix.
- Retains a classical OpenCV segmentation function in the original notebook.

The segmentation function is not connected to the classifier, and the dataset does not provide segmentation masks to validate its boundaries or cell counts.

## Dataset

The classifier uses the Blood Cell Images dataset. The image folder used for these runs is:

`dataset2-master/dataset2-master/images`

It contains four classes: EOSINOPHIL, LYMPHOCYTE, MONOCYTE, and NEUTROPHIL.

| Split | Images | Use |
|---|---:|---|
| TRAIN | 9,957 | Split into training and validation subsets |
| Training subset | 8,463 | Fit model parameters |
| Validation subset | 1,494 | Compare checkpoints |
| TEST | 2,487 | Exploratory model |

The class folders are close to balanced. Images are approximately 240 × 320 pixels. The `TEST_SIMPLE` folder is not used for the reported results.

## Results

Results below are from Google Colab with a T4 GPU. The fine-tuned checkpoint was selected using validation accuracy.

| Model | Test accuracy | Test macro-F1 |
|---|---:|---:|
| ResNet18, classifier head trained | 66.87% | 0.6658 |
| ResNet18, `layer4` fine-tuned | 79.53% | 0.7983 |

Fine-tuned model per-class test results:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| EOSINOPHIL | 0.7593 | 0.7191 | 0.7387 |
| LYMPHOCYTE | 0.9527 | 0.9097 | 0.9307 |
| MONOCYTE | 0.9204 | 0.6903 | 0.7889 |
| NEUTROPHIL | 0.6405 | 0.8622 | 0.7350 |
| Macro average | 0.8182 | 0.7953 | 0.7983 |

Validation accuracy reached 99.80%, while test accuracy was 79.53%. This difference warrants further investigation. An exact-file SHA-256 check found no duplicate images across the train, validation, and test splits; visually similar images and source-level grouping were not assessed.

The provided TEST split was used to compare the baseline and fine-tuned experiments. Treat these as prototype results, not a clinical or independent external validation.

## Setup

Install the required packages in a Python environment:

```bash
python -m pip install torch torchvision numpy scikit-learn opencv-python-headless matplotlib grad-cam kagglehub
```

Download the [Blood Cell Images dataset on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) and extract it. Set `IMAGES_ROOT` to the path to the extracted `dataset2-master/dataset2-master/images` folder. This folder should contain `TRAIN` and `TEST` subfolders, each with one folder per cell class.

## Train

In the commands below, replace `/path/to/dataset2-master/dataset2-master/images` with the actual path to your extracted images folder. Use that same path for training and evaluation.

Baseline with the ResNet18 classifier head:

```bash
python -m cell_analysis.train --images-root /path/to/dataset2-master/dataset2-master/images --epochs 5 --batch-size 32 --checkpoint models/resnet18_baseline.pt
```

Fine-tune the final ResNet18 block:

```bash
python -m cell_analysis.train --images-root /path/to/dataset2-master/dataset2-master/images --epochs 8 --batch-size 32 --checkpoint models/resnet18_layer4_best.pt --fine-tune-layer4
```

## Evaluate

Run evaluation on validation or test data:

```bash
python -m cell_analysis.evaluate --images-root/path/to/dataset2-master/dataset2-master/images --checkpoint models/resnet18_layer4_best.pt --report-dir reports/layer4-experiment --split validation
python -m cell_analysis.evaluate --images-root /path/to/dataset2-master/dataset2-master/images --checkpoint models/resnet18_layer4_best.pt --report-dir reports/layer4-experiment --split test
```

Reports include accuracy, precision, recall, F1, and a confusion matrix.

## Limitations and next steps

- The dataset consists of small cell images, not 50K × 50K whole-slide images.
- Large-image tiling, tile-boundary handling, and duplicate-object merging are not implemented.
- The watershed segmentation function is a heuristic prototype; segmentation accuracy and cell-count error have not been measured against annotated masks.
- The segmentation and classifier are separate components; the code does not yet crop watershed objects and classify each cell.
- Test results show confusion between EOSINOPHIL, MONOCYTE, and NEUTROPHIL.
- This educational prototype is not validated for clinical use.

Potential next steps are to inspect misclassified examples, evaluate on an external dataset, add annotated segmentation data, and implement memory-bounded tiled processing for large pathology images.
```