from google.colab import drive
drive.mount('/content/drive')

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def correct_labels(T, probabilities):
    """
    Correct predicted label probabilities using a given transition matrix.

    Args:
    T (np.array): A square transition matrix T of shape (n_classes, n_classes) where T[i, j]
                  is the probability of class i being mislabeled as class j.
    probabilities (np.array): An array of predicted probabilities for each class, shape (n_samples, n_classes).

    Returns:
    np.array: An array of corrected predicted labels, shape (n_samples,).

    Example usage:
    corrected_labels = correct_labels(transition_matrix, predicted_probabilities)
    """
    corrected_probs = np.linalg.inv(T).dot(probabilities.T).T
    return np.argmax(corrected_probs, axis=1)

def estimate_transition_matrix(true_labels, pred_labels):
    """
    Estimate the transition matrix from true labels to predicted labels.

    Args:
    true_labels (np.array): True labels of the data, shape (n_samples,).
    pred_labels (np.array): Predicted labels of the data, shape (n_samples,).

    Returns:
    np.array: The estimated transition matrix of shape (n_classes, n_classes).

    Example usage:
    transition_matrix = estimate_transition_matrix(y_true, y_pred)
    """
    num_classes = len(np.unique(true_labels))
    matrix = np.zeros((num_classes, num_classes))

    for t, p in zip(true_labels, pred_labels):
        matrix[t][p] += 1

    # Normalize rows to sum up to 1
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix

def evaluate(true_labels, pred_labels):
    """
    Evaluate performance metrics based on true labels and predicted labels.

    Args:
    true_labels (np.array): True labels of the data, shape (n_samples,).
    pred_labels (np.array): Predicted labels of the data, shape (n_samples,).

    Returns:
    tuple: A tuple containing the accuracy, precision, recall, and F1 score.

    Example usage:
    accuracy, precision, recall, f1 = evaluate(y_true, y_pred)
    """
    acc = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')
    f1 = f1_score(true_labels, pred_labels, average='macro')
    return acc, precision, recall, f1

def main(file_path, initial_transition_matrix=None, alpha=0.5):
    """
    Main function to train an SVM classifier on a dataset with noisy labels.

    Args:
    file_path (str): Path to the dataset file.
    initial_transition_matrix (list of lists or None): An optional initial transition matrix. If None,
                                                       a uniform matrix will be used.
    alpha (float): A parameter to blend the newly estimated transition matrix with the previous one.

    Processes:
    - Loads the dataset from the given file path.
    - Performs 10 rounds of training with a validation set to refine the transition matrix.
    - Prints the average performance metrics across the 10 rounds.
    - Trains the classifier on the entire training set and evaluates on the test set.
    - Prints the performance on the test set.

    Example usage:
    main('/content/dataset.npz', initial_transition_matrix=[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    """
    # Load data
    dataset = np.load(file_path)
    Xtr_val = dataset['Xtr'].reshape(dataset['Xtr'].shape[0], -1)
    Str_val = dataset['Str']
    Xts = dataset['Xts'].reshape(dataset['Xts'].shape[0], -1)
    Yts = dataset['Yts']

    if initial_transition_matrix is None:
        transition_matrix = np.ones((3, 3)) / 3
    else:
        transition_matrix = np.array(initial_transition_matrix)

    all_performances = []

    # Perform 10 independent training and validations
    for i in range(10):
        X_train, X_val, S_train, S_val = train_test_split(Xtr_val, Str_val, test_size=0.2)
        clf = SVC(probability=True)
        clf.fit(X_train, S_train)

        Y_pred_train = clf.predict(X_train)
        new_transition_matrix = estimate_transition_matrix(S_train, Y_pred_train)

        # Update the transition matrix
        transition_matrix = alpha * new_transition_matrix + (1 - alpha) * transition_matrix
        print(f"Estimated Transition Matrix in iteration {i+1}:\n", transition_matrix)

        S_val_probs = clf.predict_proba(X_val)
        S_val_corrected = correct_labels(transition_matrix, S_val_probs)

        Y_pred_val = clf.predict(X_val)
        acc, precision, recall, f1 = evaluate(S_val_corrected, Y_pred_val)
        all_performances.append((acc, precision, recall, f1))

    avg_performance = np.mean(all_performances, axis=0)
    print(f"Average Performance: Acc={avg_performance[0]:.4f}, Precision={avg_performance[1]:.4f}, Recall={avg_performance[2]:.4f}, F1={avg_performance[3]:.4f}")

    clf.fit(Xtr_val, Str_val)
    Y_pred_test = clf.predict(Xts)
    acc_test, precision_test, recall_test, f1_test = evaluate(Yts, Y_pred_test)
    print(f"Test Set Performance: Acc={acc_test:.4f}, Precision={precision_test:.4f}, Recall={recall_test:.4f}, F1={f1_test:.4f}")

main('/content/drive/My Drive/5328_assignment_2/datasets/FashionMNIST0.5.npz', [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])

main('/content/drive/My Drive/5328_assignment_2/datasets/FashionMNIST0.6.npz', [[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]])

main('/content/drive/My Drive/5328_assignment_2/datasets/CIFAR.npz')