# DRclassfier
A model to classify the cell stage of Deinococcus cells using fluorescence images

## Overview

This project focuses on the classification of single cells using a ResNet50 model. It encompasses a comprehensive workflow, from data pre - processing, including single cell extraction and padding, to model training and final classification. The project allows users to classify single cells into different stages (e.g., stage1 - stage5) and provides detailed classification results.

## Features

1. Data Pre - processing

   :

   - **Get ROI's masks**:Use cellpose to get the masks of target image files
   - **Single Cell Extraction**: Extract single cell images from input image data using provided masks.
   - **Image Padding**: Pad single cell images to a uniform size with zeros to ensure compatibility with the model.

2. Model Training

   :

   - **Data Augmentation**: Generate augmented versions of the dataset images to increase the diversity of the training data.
   - **Model Configuration**: Adapt the ResNet50 model for grayscale input and custom classes.
   - **Training and Evaluation**: Train the model on the augmented dataset and evaluate its performance on a test set.

3. Cell Classification

   :

   - **User - Configurable Parameters**: Allow users to set the number of channels, select the classification type (cell wall or cell membrane), and choose the input folder.
   - **Inference and Result Saving**: Classify single cells using the trained model and save the classification results to a CSV file.

## Prerequisites

- Python 3.x
- PyTorch
- NumPy
- Pandas
- Pillow
- Scikit - image
- TensorBoard
- Psutil
- CellPose2.0

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd <repository_directory>
```

1. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Data Pre - processing

Use cellpose to get the masks of target image files

To pre - process the data, run the PreProcessing.py script:

python

```python
python PreProcessing.py
```

This script will extract single cell images from the input folders and pad them to a uniform size.

### Model Training

To train the model, run the Model_training.py script:

python

```python
python Model_training.py
```

This script will log the hardware information, set up the data augmentation, create data loaders, initialize the model, and start the training process. The best model will be saved based on the test accuracy.

### Cell Classification

This is a example about how to classify DR:

The format of the input file is as follows:

Training\ExampleData\Prediction\



![](F:\OXH\typora\tmp_images\image-20250722174735310.png)

To classify single cells, run the DR_classification.py script:

python

```python
python DR_classification.py
```

This script will guide you through setting up the parameters, selecting the input folder, extracting single cells, padding the images, and finally classifying the cells. The classification results will be saved in a CSV file in the input directory.





## Project Structure

- PreProcessing.py: Contains functions for single cell extraction and padding.
- Model_training.py: Handles model training, including data augmentation, model configuration, and training loop.
- DR_classification.py: Implements the entire single cell classification workflow, from parameter setup to result saving.
- `Model/`: Contains the ResNet50 model definition.
- `runs/`: Stores the TensorBoard logs for model training.
- `Model/trained_parameters/`: Saves the trained model parameters.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.
