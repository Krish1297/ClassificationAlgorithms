import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier 
from joblib import dump, load
import numpy as np 

images_df = pd.read_csv('Images.csv', delimiter=';',skiprows=1, header=None, names=['ID', 'Class'])
edge_histogram_df = pd.read_csv('EdgeHistogram.csv', delimiter=';', skiprows=1, header=None, names=['ID', 'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10', 'Feature11', 'Feature12', 'Feature13', 'Feature14', 'Feature15', 'Feature16', 'Feature17', 'Feature18', 'Feature19', 'Feature20', 'Feature21', 'Feature22', 'Feature23', 'Feature24', 'Feature25', 'Feature26', 'Feature27', 'Feature28', 'Feature29', 'Feature30', 'Feature31', 'Feature32', 'Feature33', 'Feature34', 'Feature35', 'Feature36', 'Feature37', 'Feature38', 'Feature39', 'Feature40', 'Feature41', 'Feature42', 'Feature43', 'Feature44', 'Feature45', 'Feature46', 'Feature47', 'Feature48', 'Feature49', 'Feature50', 'Feature51', 'Feature52', 'Feature53', 'Feature54', 'Feature55', 'Feature56', 'Feature57', 'Feature58', 'Feature59', 'Feature60', 'Feature61', 'Feature62', 'Feature63', 'Feature64', 'Feature65', 'Feature66', 'Feature67', 'Feature68', 'Feature69', 'Feature70', 'Feature71', 'Feature72', 'Feature73', 'Feature74', 'Feature75', 'Feature76', 'Feature77', 'Feature78', 'Feature79', 'Feature80'])

merged_df = pd.merge(images_df, edge_histogram_df, on='ID')

merged_df.to_csv("merged_df.csv",index=False)

X = merged_df.drop(['ID', 'Class'], axis=1)
y = LabelEncoder().fit_transform(merged_df['Class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define classifiers
classifiers = {
    'K-Nearest Neighbors': KNeighborsClassifier() ,
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier()    
}

param_grids = {
    'K-Nearest Neighbors': {
        'n_neighbors': [3, 5, 7], # Number of neighbors to use
        'weights': ['uniform', 'distance'], # Weight function used in prediction
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute the nearest neighbors
        'p': [1, 2], # Power parameter for the Minkowski metric (1 for Manhattan distance, 2 for Euclidean distance)
        'leaf_size': [10, 20, 30] # Leaf size passed to BallTree or KDTree
    },
    'Support Vector Machine': {
        'kernel': ['linear', 'rbf', 'poly'], # Kernel type for the algorithm
        'C': np.logspace(-3, 2, 6), # Regularization parameter
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1], # Kernel coefficient
        'degree': [2, 3, 4] # Degree of the polynomial kernel (only for 'poly' kernel)
    },
    'Random Forest': {
        'n_estimators': [50, 100, 150], # Number of trees in the forest
        'criterion': ['gini', 'entropy'], # The function to measure the quality of a split
        'max_depth': [None, 10, 20], # The maximum depth of the trees
        'min_samples_split': [2, 5], # The minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2], # The minimum number of samples required to be at a leaf node
        'max_features': [None, 'sqrt','log2'], # The number of features to consider for the best split
    }
}

# group_number = "026"  # Replace with your actual group number

class_names = sorted(merged_df['Class'].unique())

for i, (clf_name, clf) in enumerate(classifiers.items(), start=1):
    print(f"{'='*20} {clf_name} {'='*20}")


    library = clf.__module__.split('.')[0]  # Extract the library name from the classifier's module
    test_size = X_test.shape[0] / (X_train.shape[0] + X_test.shape[0])  # Calculate test_size based on split

    # Randomized search for hyperparameter tuning
    randomized_search = RandomizedSearchCV(clf, param_distributions=param_grids[clf_name],
                                           n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
    randomized_search.fit(X_train_scaled, y_train)

    # Best hyperparameters
    best_params = randomized_search.best_params_
    print(f'Best Hyperparameters: {best_params}')
    
    # Save hyperparameters to CSV file
    param_filename = f"parameters{i}.csv"
    with open(param_filename, 'w') as file:
        file.write(f"classifier_name,{clf_name}\n")
        file.write(f"library,{library}\n")
        file.write(f"test_size,{test_size}\n") 
        for param, value in best_params.items():
            file.write(f"{param},{value}\n")

    # Training with best hyperparameters
    best_clf = randomized_search.best_estimator_
    best_clf.fit(X_train_scaled, y_train)

    # Evaluation on the test set
    predictions = best_clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    # Save confusion matrix to CSV file
    conf_matrix_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    result_filename = f"result{i}.csv"
    conf_matrix_df.to_csv(result_filename, sep=',', index_label='Class')

    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Saved Confusion Matrix to {result_filename}')
    print(f'Saved Hyperparameters to {param_filename}\n')