
# Voice Classification Model

This repository contains two Jupyter notebooks that demonstrate the process of building and evaluating machine learning models for voice classification using a neural network and a random forest classifier.

## Notebooks

### 1. Voice Classification - Initial Model

**Filename:** `voice_classification_initial.ipynb`

This notebook covers the following steps:

- **Data Import and Exploration:** 
  - The voice dataset is imported from a CSV file, and initial exploration is conducted to understand the data structure.
  - The dataset contains 5 feature columns (`V1` to `V5`) and a target column (`Class`), indicating the classification labels.

- **Data Preprocessing:**
  - The `Class` column is separated from the features to create the `X` and `y` datasets.
  - The data is split into training and testing sets using an 80/20 split ratio.

- **Neural Network Model:**
  - A Sequential model is created using TensorFlow and Keras.
  - The model consists of one hidden dense layer with 5 units and a ReLU activation function, and an output layer with a sigmoid activation function.
  - The model is compiled using binary cross-entropy as the loss function and the Adam optimizer.
  - The model is trained over 100 epochs.

- **Model Evaluation:**
  - Training accuracy and loss are plotted over the epochs.
  - The model is evaluated on the test data, showing a test accuracy of approximately 70.98%.
  - Predictions are made on the test set, and a classification report is generated to evaluate the model's precision, recall, and F1-score.

- **Random Forest Comparison:**
  - A Random Forest classifier is trained on the same dataset.
  - The model achieves a training accuracy of 100% and a test accuracy of approximately 90.75%.
  - A classification report is generated to compare the performance with the neural network model.

### 2. Voice Classification - Improved Model

**Filename:** `voice_classification_improved.ipynb`

This notebook includes similar steps to the initial model but with an important modification:

- **Data Preprocessing:**
  - The `Class` column values are modified to create a binary classification problem. Specifically, the `Class` values of 2 are changed to 0, resulting in binary classes of 0 and 1.

- **Neural Network Model:**
  - The same Sequential model architecture is used as in the initial notebook.
  - The model is trained over 100 epochs.
  - Training accuracy and loss are plotted over the epochs.

- **Model Evaluation:**
  - The model is evaluated on the test data, showing a test accuracy of approximately 80.09%.
  - Predictions are made on the test set, and a classification report is generated, showing improved performance compared to the initial model.

- **Random Forest Comparison:**
  - A Random Forest classifier is trained on the modified dataset.
  - The model achieves a training accuracy of 100% and a test accuracy of approximately 90.60%.
  - A classification report is generated to compare the performance with the neural network model.

## Conclusion

Both notebooks illustrate the process of building and evaluating machine learning models for voice classification. The improved model demonstrates a higher accuracy after fixing the target classes to create a binary classification problem. The Random Forest model consistently outperforms the neural network model, achieving higher accuracy and better classification metrics.

## Requirements

To run these notebooks, the following Python libraries are required:

- pandas
- scikit-learn
- tensorflow
- matplotlib

You can install these dependencies using pip:

```bash
pip install pandas scikit-learn tensorflow matplotlib
```

