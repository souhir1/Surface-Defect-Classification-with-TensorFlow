# Surface-Defect-Detection-with-TensorFlow-Object-Classification
This repository houses a comprehensive project aimed at tackling the challenge of surface defect detection through object classification. Utilizing the NEU Surface Defect Database from Kaggle, this project is split into two main tasks: data preparation and image classification.
# Project Overview

This project encompasses three 2 tasks, each detailed in separate Jupyter notebooks:

1. **task-1-data-preparation.ipynb**: Focused on dataset preparation, transforming raw data into the efficient TFRecord format for TensorFlow models.
2. **task2-classification.ipynb**: Implements an object classification model, detailing its architecture, optimization strategies, benchmark results and inference .
4.  **enhanced-model-complexity.ipynb**: An optional exploration that adjusts the model's complexity to address specific challenges identified in task 2.

## Task 1: Data Preparation

- Converted the dataset to tf.data.TFRecordDataset format, applying a custom feature schema for tf.train.Example.
- Generated train and test directories, each containing individual sample TFRecord files.
- The **Tfrecords-samples-for-inference** repository contains samples of the prepared data .

## Task 2: Classification

### Model Structure

- **Base Model**: Xception architecture.
- **Layers**: Includes BatchNormalization, GlobalAveragePooling2D, a dense layer with 8 units and 40% dropout, and a final dense layer for classification.
- **Optimization**: Trials with Adam, Nadam, and SGD optimizers at various learning rates, incorporating early stopping to mitigate overfitting.
- **Training**: Executed over 40 epochs with a batch size of 32, using early stopping with a patience of 15 to optimize performance.

### Benchmark Results

| Learning Rate | Adam (Accuracy, Loss) | Nadam (Accuracy, Loss) | SGD (Accuracy, Loss) |
|---------------|-----------------------|------------------------|----------------------|
| 0.001         | 95.8%, 0.1409         | 95%, 0.173             | 88%, 0.617           |
| 0.0001        | 67%, 1.2              | 65%, 0.9726            | 51%, 1.5             |
| 0.00001       | 40%, 1.654            | 38%, 1.60              | 26%, 1.7             |

The Adam optimizer at a learning rate of 0.001 yielded the best performance, with the highest test accuracy and the lowest test loss among the configurations tested.

## Enhanced Model Complexity Notebook

To tackle the rare scenario identified in task2-classification.ipynb, where validation accuracy exceeded training accuracy and validation loss was lower than training loss, due to constraints on hyperparameter adjustments set by the challenge. By creating this separate notebook, I increased the model's complexity—specifically by enhancing the dense layer to 256 units overcoming the limitations and significantly boosting the model's performance.
### Performance Comparison

| Notebook                          | Dense Layer Units | Optimizer | Learning Rate | Test Loss | Test Accuracy |
|-----------------------------------|-------------------|-----------|---------------|-----------|---------------|
| task2-classification.ipynb        | 8                 | Adam      | 0.001         | 0.1409    | 95.8%         |
| Enhanced_Model_Complexity.ipynb   | 256               | Adam      | 0.001         | 0.099     | 96%           |

Note: Despite the presence of this uncommon scenario, the original model delivered remarkable results. Both notebooks employed the Adam optimizer with a learning rate of 0.001 and implemented early stopping. The Enhanced Model Complexity notebook further refined the test data fit, evidenced by a reduced test loss, while maintaining similar accuracy levels.
