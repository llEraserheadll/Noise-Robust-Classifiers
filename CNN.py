import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score

def train_and_evaluate(Xtr_val, Str_val, Xts, Yts, transition_matrix, dataset_name):
    """
    Trains and evaluates a convolutional neural network model on a given dataset.
    
    Args:
        Xtr_val (np.array): Training and validation feature data combined.
        Str_val (np.array): Training and validation label data combined.
        Xts (np.array): Test feature data.
        Yts (np.array): Test label data.
        transition_matrix (np.array): The transition matrix for adjusting soft labels based on noise.
        dataset_name (str): The name of the dataset (used for printing results).

    Returns:
        model: The trained Keras Sequential model.
        
    This function performs the following steps:
    - One-hot encodes the training and validation labels.
    - Reshapes and prepares the test data.
    - Splits the training and validation data.
    - Applies the transition matrix to compute soft labels.
    - Builds and compiles a convolutional neural network model.
    - Trains the model using the soft labels.
    - Evaluates the model's performance on both training and test data.
    - Prints the average and standard deviation of accuracy, precision, recall, and F1 score for the test set.
    - Repeats the training and evaluation for n_iterations to average the performance metrics.
    """
    num_classes = 3  # Number of classes
    n_iterations = 10  # Number of iterations for the average performance

    # One-hot encode the labels
    encoder = OneHotEncoder(sparse=False)
    Str_val_onehot = encoder.fit_transform(Str_val.reshape(-1, 1))

    # Prepare the test data
    Xts = Xts.reshape(Xts.shape[0], Xts.shape[1], Xts.shape[2], 1)
    Yts_onehot = np.eye(num_classes)[Yts]

    # Initialize lists to store metrics for each iteration
    train_accuracies = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1s = []

    for i in range(n_iterations):
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(Xtr_val, Str_val_onehot, test_size=0.2)

        # Reshape data for convolutional network
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)

        # Compute soft labels
        soft_labels_train = np.dot(y_train, transition_matrix)
        soft_labels_val = np.dot(y_val, transition_matrix)

        # Build the model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, soft_labels_train, validation_data=(X_val, soft_labels_val), epochs=5, batch_size=32, verbose=0)

        # Get predictions for test set to calculate precision, recall, and F1-score
        y_test_pred = model.predict(Xts)
        y_test_pred_class = np.argmax(y_test_pred, axis=1)
        y_test_true_class = np.argmax(Yts_onehot, axis=1)

        test_precision = precision_score(y_test_true_class, y_test_pred_class, average='macro')
        test_recall = recall_score(y_test_true_class, y_test_pred_class, average='macro')
        test_f1 = f1_score(y_test_true_class, y_test_pred_class, average='macro')

        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_f1s.append(test_f1)

        # Evaluate the model's performance on training and test sets
        _, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        train_accuracies.append(train_accuracy)

        _, test_accuracy = model.evaluate(Xts, Yts_onehot, verbose=0)
        test_accuracies.append(test_accuracy)
    
    average_train_accuracy = np.mean(train_accuracies)
    average_test_accuracy = np.mean(test_accuracies)
    std_test_accuracy = np.std(test_accuracies)
    average_test_precision = np.mean(test_precisions)
    average_test_recall = np.mean(test_recalls)
    average_test_f1 = np.mean(test_f1s)

    print(f"Average Training Accuracy {dataset_name}: {average_train_accuracy * 100:.2f}%")
    print(f"Average Test Accuracy {dataset_name}: {average_test_accuracy * 100:.2f}%, StdDev: {std_test_accuracy * 100:.2f}%")
    print(f"Average Test Precision {dataset_name}: {average_test_precision * 100:.2f}%")
    print(f"Average Test Recall {dataset_name}: {average_test_recall * 100:.2f}%")
    print(f"Average Test F1-score {dataset_name}: {average_test_f1 * 100:.2f}%")
    return model

datasets = {
    "FashionMNIST0.5": {
        "path": 'C:/Users/abina/OneDrive - The University of Sydney (Students)/Documents/COMP5328 Assign/COMP5328_Assignment_2/FashionMNIST0.5.npz',
        "transition_matrix": np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    },
    "FashionMNIST0.6": {
        "path": 'C:/Users/abina/OneDrive - The University of Sydney (Students)/Documents/COMP5328 Assign/COMP5328_Assignment_2/FashionMNIST0.6.npz',
        "transition_matrix": np.array([[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]])  
    },
}

for dataset_name, dataset_info in datasets.items():
    dataset = np.load(dataset_info['path'])
    Xtr_val = dataset['Xtr']
    Str_val = dataset['Str']
    Xts = dataset['Xts']
    Yts = dataset['Yts']
    model = train_and_evaluate(Xtr_val, Str_val, Xts, Yts, dataset_info['transition_matrix'], dataset_name)

def train_and_evaluate(Xtr_val_cifar, Str_val_cifar, Xts_cifar, Yts_cifar, transition_matrix_cifar):
    """
    Train and evaluate a Convolutional Neural Network (CNN) on the CIFAR dataset.

    This function takes the training/validation and testing datasets along with
    a transition matrix to handle label noise during training. It trains a CNN
    using the training subset, evaluates it on the validation subset in each
    iteration, and finally reports the performance on the test set. The training
    is repeated for a number of iterations to assess the stability of the model's
    performance. The function reports the average and standard deviation of
    accuracies, as well as precision, recall, and F1-score on the test set.

    Parameters:
    Xtr_val_cifar (np.array): Numpy array containing the training/validation features.
    Str_val_cifar (np.array): Numpy array containing the noisy training/validation labels.
    Xts_cifar (np.array): Numpy array containing the test features.
    Yts_cifar (np.array): Numpy array containing the true test labels.
    transition_matrix_cifar (np.array): A matrix representing the probabilities of
                                        transitioning from true labels to noisy labels.

    Returns:
    model_cifar (Sequential): The trained Keras Sequential CNN model.

    Prints:
    - Average Training Accuracy for CIFAR: The average accuracy on the training set over the iterations.
    - Average Test Accuracy for CIFAR: The average accuracy on the test set over the iterations.
    - StdDev Test Accuracy for CIFAR: The standard deviation of the test set accuracies over the iterations.
    - Average Test Precision for CIFAR: The average macro precision on the test set over the iterations.
    - Average Test Recall for CIFAR: The average macro recall on the test set over the iterations.
    - Average Test F1-score for CIFAR: The average macro F1-score on the test set over the iterations.

    Example usage:
    model = train_and_evaluate(Xtr_val, Str_val, Xts, Yts, T_matrix)
    """
    num_classes_cifar = 3
    n_iterations_cifar = 10

    encoder_cifar = OneHotEncoder(sparse=False)
    Str_val_onehot_cifar = encoder_cifar.fit_transform(Str_val_cifar.reshape(-1, 1))
    Yts_onehot_cifar = encoder_cifar.transform(Yts_cifar.reshape(-1, 1))

    train_accuracies_cifar = []
    test_accuracies_cifar = []
    test_precisions_cifar = []
    test_recalls_cifar = []
    test_f1s_cifar = []

    for _ in range(n_iterations_cifar):
        X_train_cifar, X_val_cifar, y_train_cifar, y_val_cifar = train_test_split(Xtr_val_cifar, Str_val_onehot_cifar, test_size=0.2)
        X_train_cifar = X_train_cifar.reshape(X_train_cifar.shape[0], 32, 32, 3)
        X_val_cifar = X_val_cifar.reshape(X_val_cifar.shape[0], 32, 32, 3)
        Xts_cifar = Xts_cifar.reshape(Xts_cifar.shape[0], 32, 32, 3)

        soft_labels_train_cifar = np.dot(y_train_cifar, transition_matrix_cifar)
        soft_labels_val_cifar = np.dot(y_val_cifar, transition_matrix_cifar)

        model_cifar = Sequential()
        model_cifar.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
        model_cifar.add(MaxPooling2D(pool_size=(2, 2)))
        model_cifar.add(Flatten())
        model_cifar.add(Dense(128, activation='relu'))
        model_cifar.add(Dense(num_classes_cifar, activation='softmax'))

        model_cifar.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model_cifar.fit(X_train_cifar, soft_labels_train_cifar, validation_data=(X_val_cifar, soft_labels_val_cifar), epochs=5, batch_size=32, verbose=0)

        _, train_accuracy_cifar = model_cifar.evaluate(X_train_cifar, y_train_cifar, verbose=0)
        train_accuracies_cifar.append(train_accuracy_cifar)

        _, test_accuracy_cifar = model_cifar.evaluate(Xts_cifar, Yts_onehot_cifar, verbose=0)
        test_accuracies_cifar.append(test_accuracy_cifar)

        y_test_pred_cifar = model_cifar.predict(Xts_cifar)
        y_test_pred_class_cifar = np.argmax(y_test_pred_cifar, axis=1)
        y_test_true_class_cifar = np.argmax(Yts_onehot_cifar, axis=1)

        test_precision_cifar = precision_score(y_test_true_class_cifar, y_test_pred_class_cifar, average='macro')
        test_recall_cifar = recall_score(y_test_true_class_cifar, y_test_pred_class_cifar, average='macro')
        test_f1_cifar = f1_score(y_test_true_class_cifar, y_test_pred_class_cifar, average='macro')

        test_precisions_cifar.append(test_precision_cifar)
        test_recalls_cifar.append(test_recall_cifar)
        test_f1s_cifar.append(test_f1_cifar)

    print(f"Average Training Accuracy for CIFAR: {np.mean(train_accuracies_cifar) * 100:.2f}%")
    print(f"Average Test Accuracy for CIFAR: {np.mean(test_accuracies_cifar) * 100:.2f}%")
    print(f"StdDev Test Accuracy for CIFAR: {np.std(test_accuracies_cifar) * 100:.2f}%")
    print(f"Average Test Precision for CIFAR: {np.mean(test_precisions_cifar) * 100:.2f}%")
    print(f"Average Test Recall for CIFAR: {np.mean(test_recalls_cifar) * 100:.2f}%")
    print(f"Average Test F1-score for CIFAR: {np.mean(test_f1s_cifar) * 100:.2f}%")

    return model_cifar

# Estimation of the transition matrix
def estimate_transition_matrix_cifar(true_labels_cifar, predicted_labels_cifar):
    """
    Estimates the transition matrix for CIFAR dataset based on true and predicted labels.
    
    Args:
        true_labels_cifar (np.array): The true labels of the CIFAR dataset.
        predicted_labels_cifar (np.array): The predicted labels of the CIFAR dataset by the model.

    Returns:
        np.array: The estimated transition matrix derived from the confusion matrix.
    
    The transition matrix represents the probability of predicting a class given the true class label. 
    It is calculated by normalizing each row of the confusion matrix to sum to 1.
    """
    conf_mat_cifar = confusion_matrix(true_labels_cifar, predicted_labels_cifar)
    transition_matrix_cifar = conf_mat_cifar / conf_mat_cifar.sum(axis=1, keepdims=True)
    return transition_matrix_cifar

# Load CIFAR dataset
dataset_path_cifar = 'C:/Users/abina/OneDrive - The University of Sydney (Students)/Documents/COMP5328 Assign/COMP5328_Assignment_2/CIFAR.npz'
dataset_cifar = np.load(dataset_path_cifar)
Xtr_val_cifar = dataset_cifar['Xtr']
Str_val_cifar = dataset_cifar['Str']
Xts_cifar = dataset_cifar['Xts']
Yts_cifar = dataset_cifar['Yts']

model_cifar = train_and_evaluate(Xtr_val_cifar, Str_val_cifar, Xts_cifar, Yts_cifar, np.eye(3))

predicted_labels_cifar = model_cifar.predict(Xts_cifar)
predicted_labels_cifar = np.argmax(predicted_labels_cifar, axis=1)
transition_matrix_cifar = estimate_transition_matrix_cifar(Yts_cifar, predicted_labels_cifar)

print(f"Estimated Transition Matrix for CIFAR:")
print(transition_matrix_cifar)