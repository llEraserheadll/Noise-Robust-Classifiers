import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


"""
--- FashionMNIST0.5 and FashionMNIST0.6 DATASET ---
"""

# Remember to replace the FILE_PATH to load datasets
dataset_MNIST5 = np.load ('/Users/hillarymulyadi/Documents/Masters degree/Semester 2/COMP5328 - Advanced Machine Learning/Assignment_2/FashionMNIST0.5.npz')
dataset_MNIST6 = np.load ('/Users/hillarymulyadi/Documents/Masters degree/Semester 2/COMP5328 - Advanced Machine Learning/Assignment_2/FashionMNIST0.6.npz')

# Check the available keys in the dataset
print("Available keys in the FashionMNIST0.5 dataset:", list(dataset_MNIST5.keys()))
print("Available keys in the FashionMNIST0.6 dataset:", list(dataset_MNIST6.keys()))

# Features of the training data
Xtr_MNIST5 = dataset_MNIST5['Xtr']
Xtr_MNIST6 = dataset_MNIST6['Xtr']

# Contains noisy labels
Str_MNIST5 = dataset_MNIST5['Str']
Str_MNIST6 = dataset_MNIST6['Str']

# Contains features of the test data
Xts_MNIST5 = dataset_MNIST5['Xts']
Xts_MNIST6 = dataset_MNIST6['Xts']

# Contains clean labels of the test data
Yts_MNIST5 = dataset_MNIST5['Yts']
Yts_MNIST6 = dataset_MNIST6['Yts']

print(Xtr_MNIST5.shape)
print(Str_MNIST5.shape)
print(Xts_MNIST5.shape)
print(Yts_MNIST5.shape)

print(Xtr_MNIST6.shape)
print(Str_MNIST6.shape)
print(Xts_MNIST6.shape)
print(Yts_MNIST6.shape)

# FashionMNIST0.5
## Split Xtr and Str into random train and val subsets 80/20.
from sklearn.model_selection import train_test_split
X_train_MNIST5, X_val_MNIST5, y_train_MNIST5, y_val_MNIST5 = train_test_split(Xtr_MNIST5, Str_MNIST5, test_size=0.20, random_state=42)
print(X_train_MNIST5.shape) #Xtr - features
print(X_val_MNIST5.shape)
print(y_train_MNIST5.shape) #Str - labels
print(y_val_MNIST5.shape)

# FashionMNIST0.5
## Split Xtr and Str into random train and val subsets 80/20.
from sklearn.model_selection import train_test_split
X_train_MNIST6, X_val_MNIST6, y_train_MNIST6, y_val_MNIST6 = train_test_split(Xtr_MNIST6, Str_MNIST6, test_size=0.20, random_state=42)
print(X_train_MNIST6.shape) #Xtr - features
print(X_val_MNIST6.shape)
print(y_train_MNIST6.shape) #Str - labels
print(y_val_MNIST6.shape)

# Corrected labels
def correct_labels(noisy_labels, transition_matrix):
    corrected_labels = np.argmax(transition_matrix[noisy_labels, :], axis=1)
    return corrected_labels

# FashionMNIST0.5
plt.figure(figsize=(20, 2))
plt.suptitle("Fashion MNIST 0.5 samples", size=16)
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_train_MNIST5[i, :], cmap=plt.cm.gray, aspect='auto')
    plt.xticks(())
    plt.yticks(())

# FashionMNIST0.6
plt.figure(figsize=(20, 2))
plt.suptitle("Fashion MNIST 0.6 samples", size=16)
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_train_MNIST6[i, :], cmap=plt.cm.gray, aspect='auto')
    plt.xticks(())
    plt.yticks(())

# FashionMNIST0.5
## Reshaping the X_train to flatten it to 2D
X_train_MNIST5 = X_train_MNIST5.reshape(X_train_MNIST5.shape[0], -1)
print('X_train_MNIST5 shape is', X_train_MNIST5.shape)

## Reshaping the X_val to flatten it to 2D
X_val_MNIST5 = X_val_MNIST5.reshape(X_val_MNIST5.shape[0], -1)
print('X_val_MNIST5 shape is', X_val_MNIST5.shape)

## Reshaping the X_train to flatten it to 2D
Xts_MNIST5 = Xts_MNIST5.reshape(Xts_MNIST5.shape[0], -1)
print('Xts_MNIST5 shape is', Xts_MNIST5.shape)

# FashionMNIST0.6
## Reshaping the X_train to flatten it to 2D
X_train_MNIST6 = X_train_MNIST6.reshape(X_train_MNIST6.shape[0], -1)
print('X_train_MNIST6 shape is', X_train_MNIST6.shape)

## Reshaping the X_val to flatten it to 2D
X_val_MNIST6 = X_val_MNIST6.reshape(X_val_MNIST6.shape[0], -1)
print('X_val_MNIST6 shape is', X_val_MNIST6.shape)

## Reshaping the X_train to flatten it to 2D
Xts_MNIST6 = Xts_MNIST6.reshape(Xts_MNIST6.shape[0], -1)
print('Xts_MNIST6 shape is', Xts_MNIST6.shape)

# FashionMNIST0.5
## Correct the labels in y_train and y_val
transition_matrix_MNIST5 = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
y_train_corrected_MNIST5 = correct_labels(y_train_MNIST5, transition_matrix_MNIST5)
y_val_corrected_MNIST5 = correct_labels(y_val_MNIST6, transition_matrix_MNIST5)

# FashionMNIST0.6
transition_matrix_MNIST6 = np.array([[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]])
y_train_corrected_MNIST6 = correct_labels(y_train_MNIST6, transition_matrix_MNIST6)
y_val_corrected_MNIST6 = correct_labels(y_val_MNIST6, transition_matrix_MNIST6)


"""
--- Grid-search with 3-fold cross validation for hyperparameter tuning---
"""

param_grid = {'n_estimators': [100, 200, 300],
              'max_samples': [100, 250, 400]}


grid_search_MNIST5 = GridSearchCV(BaggingClassifier(DecisionTreeClassifier(criterion='entropy', random_state=42), bootstrap = True, random_state=42), param_grid, cv=3, return_train_score=True)

grid_search_MNIST6 = GridSearchCV(BaggingClassifier(DecisionTreeClassifier(criterion='entropy', random_state=42), bootstrap = True, random_state=42), param_grid, cv=3, return_train_score=True)

grid_search_MNIST5.fit(X_train_MNIST5, y_train_corrected_MNIST5)

grid_search_MNIST6.fit(X_train_MNIST6, y_train_corrected_MNIST6)

# FashionMNIST0.5
print("Validation set score: {:.2f}".format(grid_search_MNIST5.score(X_val_MNIST5, y_val_corrected_MNIST5))) #give us score on the best parameters
print("Best parameters: {}".format(grid_search_MNIST5.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search_MNIST5.best_score_)) #prints the average score of the best model across all cross-validation folds
print("Best estimator:\n{}".format(grid_search_MNIST5.best_estimator_))

# FashionMNIST0.6
print("Validation set score: {:.2f}".format(grid_search_MNIST6.score(X_val_MNIST6, y_val_corrected_MNIST6))) #give us score on the best parameters
print("Best parameters: {}".format(grid_search_MNIST6.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search_MNIST6.best_score_)) #prints the average score of the best model across all cross-validation folds
print("Best estimator:\n{}".format(grid_search_MNIST6.best_estimator_))

"""
--- Training and Evaluation with best parameters---
"""
# FashionMNIST5
## Using the best parameters found from grid-search during run
clf_MNIST5 = BaggingClassifier(estimator=DecisionTreeClassifier(criterion='entropy'), n_estimators=300, max_samples=400).fit(X_train_MNIST5, y_train_corrected_MNIST5)

# FashionMNIST6
## Using the best parameters found from grid-search during run
clf_MNIST6 = BaggingClassifier(estimator=DecisionTreeClassifier(criterion='entropy'), n_estimators=400, max_samples=200).fit(X_train_MNIST6, y_train_corrected_MNIST6)

# Evaluating with Recall, Precision, F1, and Top-1 Accuracy
## To have a rigorous performance evaluation, you need to train each classifier at least 10 times
## with the different training and validation sets generated by random sampling. Then report both
## the mean and the standard derivation of each metric.
cv = KFold(n_splits=10, shuffle=True)

# FashionMNIST0.5
y_pred_MNIST5 = clf_MNIST5.predict(Xts_MNIST5)
accuracy_scores_MNIST5 = cross_val_score(clf_MNIST5, Xts_MNIST5, Yts_MNIST5, cv=cv, scoring='accuracy')
recall_scores_MNIST5 = cross_val_score(clf_MNIST5, Xts_MNIST5, Yts_MNIST5, cv=cv, scoring='recall_macro')
precision_scores_MNIST5 = cross_val_score(clf_MNIST5, Xts_MNIST5, Yts_MNIST5, cv=cv, scoring='precision_macro')
f1_scores_MNIST5 = cross_val_score(clf_MNIST5, Xts_MNIST5, Yts_MNIST5, cv=cv, scoring='f1_macro')

print(f"Top-1 Accuracy - Cross-validation mean scores: {np.mean(accuracy_scores_MNIST5):.4f}")
print(f"Top-1 Accuracy - Cross-validation scores standard deviation: {np.std(accuracy_scores_MNIST5):.4f}")
print(f"Recall - Cross-validation mean scores: {np.mean(recall_scores_MNIST5):.4f}")
print(f"Recall - Cross-validation scores standard deviation: {np.std(recall_scores_MNIST5):.4f}")
print(f"Precision - Cross-validation scores mean scores: {np.mean(precision_scores_MNIST5):.4f}")
print(f"Precision - Cross-validation scores standard deviation: {np.std(precision_scores_MNIST5):.4f}")
print(f"F1 Score - Cross-validation scores mean scores: {np.mean(f1_scores_MNIST5):.4f}")
print(f"F1 Score - Cross-validation scores standard deviation: {np.std(f1_scores_MNIST5):.4f}")

# FashionMNIST0.6
y_pred_MNIST6 = clf_MNIST6.predict(Xts_MNIST6)
accuracy_scores_MNIST6 = cross_val_score(clf_MNIST6, Xts_MNIST6, Yts_MNIST6, cv=cv, scoring='accuracy')
recall_scores_MNIST6 = cross_val_score(clf_MNIST6, Xts_MNIST6, Yts_MNIST6, cv=cv, scoring='recall_macro')
precision_scores_MNIST6 = cross_val_score(clf_MNIST6, Xts_MNIST6, Yts_MNIST6, cv=cv, scoring='precision_macro')
f1_scores_MNIST6 = cross_val_score(clf_MNIST6, Xts_MNIST6, Yts_MNIST6, cv=cv, scoring='f1_macro')

print(f"Top-1 Accuracy - Cross-validation mean scores: {np.mean(accuracy_scores_MNIST6):.4f}")
print(f"Top-1 Accuracy - Cross-validation scores standard deviation: {np.std(accuracy_scores_MNIST6):.4f}")
print(f"Recall - Cross-validation mean scores: {np.mean(recall_scores_MNIST6):.4f}")
print(f"Recall - Cross-validation scores standard deviation: {np.std(recall_scores_MNIST6):.4f}")
print(f"Precision - Cross-validation scores mean scores: {np.mean(precision_scores_MNIST6):.4f}")
print(f"Precision - Cross-validation scores standard deviation: {np.std(precision_scores_MNIST6):.4f}")
print(f"F1 Score - Cross-validation scores mean scores: {np.mean(f1_scores_MNIST6):.4f}")
print(f"F1 Score - Cross-validation scores standard deviation: {np.std(f1_scores_MNIST6):.4f}")


"""
--- CIFAR DATASET ---
"""

# Remember to replace the FILE_PATH to load datasets
dataset_CIFAR = np.load ('/Users/hillarymulyadi/Documents/Masters degree/Semester 2/COMP5328 - Advanced Machine Learning/Assignment_2/CIFAR.npz')

# Check the available keys in the dataset
print("Available keys in the CIFAR dataset:", list(dataset_CIFAR.keys()))

# Features of the training data
Xtr_CIFAR = dataset_CIFAR['Xtr']

# Contains noisy labels
Str_CIFAR = dataset_CIFAR['Str']

# Contains features of the test data
Xts_CIFAR = dataset_CIFAR['Xts']

# Contains clean labels of the test data
Yts_CIFAR = dataset_CIFAR['Yts']

print(Xtr_CIFAR.shape)
print(Str_CIFAR.shape)
print(Xts_CIFAR.shape)
print(Yts_CIFAR.shape)

## Split Xtr and Str into random train and val subsets 80/20.
from sklearn.model_selection import train_test_split
X_train_CIFAR, X_val_CIFAR, y_train_CIFAR, y_val_CIFAR = train_test_split(Xtr_CIFAR, Str_CIFAR, test_size=0.20, random_state=42)
print(X_train_CIFAR.shape) #Xtr - features
print(X_val_CIFAR.shape)
print(y_train_CIFAR.shape) #Str - labels
print(y_val_CIFAR.shape)
print(Xts_CIFAR.shape)
print(Yts_CIFAR.shape)

# CIFAR Sample Images
plt.figure(figsize=(20, 2))
plt.suptitle("CIFAR samples", size=10)
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_train_CIFAR[i, :], cmap=plt.cm.gray, aspect='auto')
    plt.xticks(())
    plt.yticks(())

# Reshaping the X_train to flatten it to 2D
X_train_CIFAR = X_train_CIFAR.reshape(X_train_CIFAR.shape[0], -1)
print('X_train_CIFAR shape is', X_train_CIFAR.shape)

# Reshaping the X_val to flatten it to 2D
X_val_CIFAR = X_val_CIFAR.reshape(X_val_CIFAR.shape[0], -1)
print('X_val_CIFAR shape is', X_val_CIFAR.shape)

# Reshaping the X_train to flatten it to 2D
Xts_CIFAR = Xts_CIFAR.reshape(Xts_CIFAR.shape[0], -1)
print('Xts_CIFAR shape is', Xts_CIFAR.shape)

"""
--- Grid-search with 3-fold cross validation for hyperparameter tuning ---
"""

param_grid = {'n_estimators': [100, 200, 300],
              'max_samples': [100, 250, 400]}

grid_search_CIFAR = GridSearchCV(BaggingClassifier(DecisionTreeClassifier(criterion='entropy', random_state=42), bootstrap = True, random_state=42), param_grid, cv=3,
                          return_train_score=True)

grid_search_CIFAR.fit(X_train_CIFAR, y_train_CIFAR)

print("Validation set score: {:.2f}".format(grid_search_CIFAR.score(X_val_CIFAR, y_val_CIFAR))) #give us score on the best parameters
print("Best parameters: {}".format(grid_search_CIFAR.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search_CIFAR.best_score_)) #prints the average score of the best model across all cross-validation folds
print("Best estimator:\n{}".format(grid_search_CIFAR.best_estimator_))

"""
--- Training and Evaluation with best parameters ---
"""
# Using the best parameters found from grid-search during run
clf_CIFAR = BaggingClassifier(estimator=DecisionTreeClassifier(criterion='entropy'), n_estimators=300, max_samples=250).fit(X_train_CIFAR, y_train_CIFAR)

"""
--- Transition Matrix Estimation ---
"""

num_classes = len(np.unique(y_train_CIFAR)) #3
probs = clf_CIFAR.predict_proba(X_train_CIFAR)
argmax_probs = np.argmax(probs, axis=1)

num_classes_test = np.max(argmax_probs) + 1

# Initialize the transition matrix with zeros
transition_matrix = np.zeros((num_classes, num_classes))

# Estimate the transition matrix
for argmax_probs, y_train_CIFAR in zip(argmax_probs, y_train_CIFAR):
    transition_matrix[argmax_probs, y_train_CIFAR] += 1

# Normalize the transition matrix to get probabilities
transition_matrix /= np.sum(transition_matrix, axis=0)

print(transition_matrix)

"""
--- Correct labels with estimated transition matrix ---
"""
# CIFAR with estimated transition matrix
y_train_corrected_CIFAR = correct_labels(y_train_CIFAR, transition_matrix)
y_val_corrected_CIFAR = correct_labels(y_val_CIFAR, transition_matrix)

"""
--- Training and Evaluation with best parameters using estimated transition matrix---
"""

# Train classifier on corrected labels with estimated transition matrix
clf_CIFAR = BaggingClassifier(estimator=DecisionTreeClassifier(criterion='entropy'), n_estimators=grid_search_CIFAR.best_params_['n_estimators'], max_samples=grid_search_CIFAR.best_params_['max_samples']).fit(X_train_CIFAR, y_train_corrected_CIFAR)

cv = KFold(n_splits=10, shuffle=True)
y_pred_CIFAR = clf_CIFAR.predict(Xts_CIFAR)

accuracy_scores_CIFAR = cross_val_score(clf_CIFAR, Xts_CIFAR, Yts_CIFAR, cv=cv, scoring='accuracy')
recall_scores_CIFAR = cross_val_score(clf_CIFAR, Xts_CIFAR, Yts_CIFAR, cv=cv, scoring='recall_macro')
precision_scores_CIFAR = cross_val_score(clf_CIFAR, Xts_CIFAR, Yts_CIFAR, cv=cv, scoring='precision_macro')
f1_scores_CIFAR = cross_val_score(clf_CIFAR, Xts_CIFAR, Yts_CIFAR, cv=cv, scoring='f1_macro')

print(f"Top-1 Accuracy - Cross-validation mean scores: {np.mean(accuracy_scores_CIFAR):.4f}")
print(f"Top-1 Accuracy - Cross-validation scores standard deviation: {np.std(accuracy_scores_CIFAR):.4f}")
print(f"Recall - Cross-validation mean scores: {np.mean(recall_scores_CIFAR):.4f}")
print(f"Recall - Cross-validation scores standard deviation: {np.std(recall_scores_CIFAR):.4f}")
print(f"Precision - Cross-validation scores mean scores: {np.mean(precision_scores_CIFAR):.4f}")
print(f"Precision - Cross-validation scores standard deviation: {np.std(precision_scores_CIFAR):.4f}")
print(f"F1 Score - Cross-validation scores mean scores: {np.mean(f1_scores_CIFAR):.4f}")
print(f"F1 Score - Cross-validation scores standard deviation: {np.std(f1_scores_CIFAR):.4f}")


