# Image Segmentation with UNet++ and EfficientNet

This project focuses on image segmentation, specifically for segmenting humans in various environments and settings. The goal is to generate accurate masks for human figures in images using a deep learning model. The project leverages the **UNet++ architecture** with **EfficientNet** as the encoder, trained on a dataset of images and corresponding masks.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
<!--- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)-->

## Overview
Key features of the project include:
- **Dataset**: Images of humans in diverse environments with corresponding masks.
- **Data Augmentation**: Enhanced the data using the `albumentations` library.
- **Model**: UNet++ architecture with EfficientNet as the encoder, pretrained on ImageNet.
- **Training**: Trained using a combination of Dice Loss and Binary Cross Entropy, with Adam as the optimizer.
- **Performance**: Significant reduction in validation loss achieved in just 10 epochs.

## Dataset
The dataset consists of:
- **Images**: RGB images of humans in various environments.
- **Masks**: Corresponding binary masks for human segmentation.

The dataset is preprocessed and split into training and validation sets for model training.

## Data Preprocessing
To improve model generalization and robustness, the following steps were taken:
1. **Data Augmentation**: Applied using the `albumentations` library. Techniques include:
   - Random cropping
   - Horizontal flipping
   - Rotation
   - Brightness/contrast adjustments
2. **Preprocessing**:
   - Images and masks are resized and reshaped to `(Channels, Height, Width)` format supported by PyTorch.
   - Normalization: Pixel values are scaled to the range `[0, 1]` by dividing by 255.

## Model Architecture
The model is based on the **UNet++** architecture, which is an extension of the classic UNet model. Key components include:
- **Encoder**: EfficientNet (pretrained on ImageNet) is used as the encoder to extract feature maps.
- **Decoder**: The decoder path in UNet++ is designed to capture fine-grained details for accurate segmentation.
- **Input**: 3-channel RGB images.
- **Output**: Binary masks for human segmentation.

## Training
The model was trained with the following configuration:
- **Loss Function**: A combination of **Dice Loss** and **Binary Cross Entropy** to handle class imbalance and improve segmentation accuracy.
- **Optimizer**: Adam optimizer with a learning rate of `0.001`.
- **Batch Size**: 32.
- **Epochs**: 10.

## Results
- **Validation Loss**: A significant reduction in validation loss was observed within 10 epochs, demonstrating the model's ability to learn effectively.
- **Performance**: The model achieves high accuracy in generating segmentation masks.

## Credits
Credits to Vikram Shenoy for providing the dataset. https://github.com/VikramShenoy97/Human-Segmentation-Dataset
