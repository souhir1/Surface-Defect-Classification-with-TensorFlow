# Surface-Defect-Classification-with-TensorFlow
This repository houses a comprehensive project aimed at tackling the challenge of surface defect classification

Utilizing the NEU Surface Defect Database from Kaggle, this project encompasses two tasks, each detailed in separate Jupyter notebooks:

1. **task-1-data-preparation.ipynb**: Focused on dataset preparation, transforming raw data into the efficient TFRecord format for TensorFlow models.
2. **task2-classification.ipynb**: Details the fine-tuning of the Xception model for Surface Defect Classification using TensorFlow, covering its architectural configuration, optimization techniques, benchmark outcomes and inference process.
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

The Adam optimizer, set at a learning rate of 0.001, delivered optimal results, achieving the highest test accuracy and the lowest test loss among all tested configurations. Consequently, it has been saved and is utilized for inference.
## Enhanced Model Complexity Notebook

Despite the impressive performance of the model in task2-classification.ipynb, I encountered an unusual scenario where the validation accuracy surpassed the training accuracy. This was due to the simplistic design of the dense layer, which comprised only 8 units and had a 40% dropout rate. Limited by these hyperparameter settings due to the challenge's constraints, I tackled this issue in enhanced-model-complexity.ipynb by increasing the dense layer to 256 units, which improved the model's performance. 
### Performance Comparison

| Notebook                          | Dense Layer Units | Optimizer | Learning Rate | Test Loss | Test Accuracy |
|-----------------------------------|-------------------|-----------|---------------|-----------|---------------|
| task2-classification.ipynb        | 8                 | Adam      | 0.001         | 0.1409    | 95.8%         |
| Enhanced_Model_Complexity.ipynb   | 256               | Adam      | 0.001         | 0.099     | 96%           |

Note: Even with the occurrence of this unusual scenario, the original model achieved remarkable results. Both notebooks utilized the Adam optimizer with a learning rate of 0.001 . The enhanced-model-complexity notebook further refined the test data fit, evidenced by a reduced test loss, while maintaining similar accuracy levels.
