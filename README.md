# Image classification using K-Nearest Neighbors, Support Vector Machine, and Random Forest classifiers

This repository contains a machine learning project for image classification using K-Nearest Neighbors, Support Vector Machine, and Random Forest classifiers. The project involves preprocessing image features, splitting the dataset, hyperparameter tuning, and evaluating the classifiers.

### Files

1. `Images.csv` CSV file containing image IDs and corresponding classes.
2. `EdgeHistogram.csv` CSV file containing edge histogram features for each image ID.
3. `merged_df.csv` Merged dataset combining information from `Images.csv` and `EdgeHistogram.csv`.
4. `Solution.ipynb` Python script for preprocessing, model training, hyperparameter tuning, and evaluation.
5. `parameters1.csv`, `parameters2.csv`, `parameters3.csv` CSV files containing the best hyperparameters for each classifier.
6. `result1.csv`, `result2.csv`, `result3.csv` CSV files containing confusion matrices for each classifier on the test set.
7. `list_of_hyperparameters_used.csv` - CSV files containing the list of all used Hyperparameters for the 3 different classifiers.
8. `readme.txt` This file, providing instructions and information about the project.

### Instructions

1. Dependencies
     ```
   - Install the required Python libraries by executing the following command
     ```
     pip install pandas scikit-learn joblib numpy
     ```

2. Dataset
   - Place `Images.csv` and `EdgeHistogram.csv` in the same directory.

3. Data Preparation
   - The notebook preprocesses and merges the image datasets into `merged_df.csv`.

4. Training and Evaluation
   - The notebook uses three classifiers K-Nearest Neighbors, Support Vector Machine, and Random Forest.
   - Hyperparameters for each classifier are tuned using a randomized search with cross-validation.
   - Best hyperparameters, test accuracy, and confusion matrices are saved in respective CSV files.

### Output Files

1. `parameters1.csv`, `parameters2.csv`, `parameters3.csv`
   - CSV files containing the best hyperparameters for each classifier.

2. `result1.csv`, `result2.csv`, `result3.csv`
   - CSV files containing confusion matrices for each classifier on the test set.

