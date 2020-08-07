import numpy as np
from collections import Counter

def accuracy(y_true, y_pred):
    """
    Function to calculate accuracy

    Args:
        y_true: list of true values
        y_pred: list of pedicted values
    """

    # initialize a simple counter for correct predictions
    correct_coounter = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct_coounter += 1
    return correct_coounter / len(y_true)


def true_positive(y_true, y_pred):
    """
    Function to calculate true positives
    
    Args:
        y_true: list of true values
        y_pred: list of predicted values
    """
    
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    return tp


def true_negative(y_true, y_pred):
    """
    Calculates the true negatives

    Args:
        y_true: list of true values
        y_pred: list of predicted values
    """

    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn


def false_positive(y_true, y_pred):
    """
    Calculates the false positives

    Args:
        y_true: list of true values
        y_pred: list of predicted values
    """

    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp


def false_negative(y_true, y_pred):
    """
    Calculates the false negatives

    Args:
        y_true: list of true values
        y_pred: list of predicted values
    """

    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    return fn


def accuracy_v2(y_true, y_pred):
    """
    Function that calculates accuracy using tp/tn/fp/fn

    Args:
        y_true: list of true values
        y_pred: list of predicted values
    """

    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)

    accuraccy_score = (tp + tn) / (tp + tn + fp + fn)

    return accuraccy_score


def precision(y_true, y_pred):
    """
    Function to calculate precision

    Args:
        y_true: list of true values
        y_pred: list of predicted values
    """

    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    precision = tp / (tp + fp)
    return precision


def recall(y_true, y_pred):
    """
    Function to calculate recall

    Args:
        y_true: list of true values
        y_pred: list of predicted values
    """

    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    recall = tp / (tp + fn)
    return recall


def f1(y_true, y_pred):
    """
    Function to calculate f1

    Args:
        y_true: list of true values
        y_pred: list of predicted values
    """

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f1 = 2 * p * r / (p + r)
    return f1


def tpr(y_true, y_pred):
    """
    Function to calculate tpr

    Args:
        y_true: list of true values
        y_pred: list of predicted values
    """

    return recall(y_true, y_pred)


def fpr(y_true, y_pred):
    """
    Function to calculate fpr

    Args:
        y_true: list of true values
        y_pred: list of predicted values
    """

    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)

    return fp / (tn + fp)


def log_loss(y_true, y_proba):
    """
    Function to calculate log loss

    Args:
        y_true: list of true values
        y_proba: list of probalilities
    """
    
    # define an epsilon value
    # this value is used to clip the probabilities
    epsilon = 1e-15
    # initialize empty list to store inidivual losses
    loss = []
    for yt, yp in zip(y_true, y_proba):
        # adjust probabilities
        # 0 gets converted to 1e-15
        # 1 gets converted to 1 - 1e-15
        yp = np.clip(yp, epsilon, 1 - epsilon)
        # calculate loss for one sample
        temp_loss = -1.0 * (
            yt * np.log(yp)
            + (1 - yt) * np.log(1 - yp)
        )
        # add the loss to the list
        loss.append(temp_loss)
    return np.mean(loss)



def macro_precision(y_true, y_pred):
    """
    Function to calculate macro precision

    Args:
        y_true: list of true values
        y_pred: list of predicted values
    """

    # find the number of classes
    num_classes = len(np.unique(y_true))

    # initialize precision at 0
    precision = 0

    for class_ in range(num_classes):
        # all classes except the current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive for current class
        tp = true_positive(temp_true, temp_pred)

        # calculate false positive for current class
        fp = false_positive(temp_true, temp_pred)

        temp_precision = tp / (tp + fp)

        precision += temp_precision
    
    # calculate average over all classes
    precision /= num_classes
    return precision


def micro_precision(y_true, y_pred):
    """
    Function to calculate micro precision

    Args:
        y_true: list of true values
        y_pred: list of predicted values
    """

    # find the number of classes
    num_classes = len(np.unique(y_true))

    # initialize tp and fp to 0
    tp = 0
    fp = 0

    for class_ in range(num_classes):
        # all classes except the current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive for current class
        tp += true_positive(temp_true, temp_pred)

        # calculate false positive for current class
        fp += false_positive(temp_true, temp_pred)
    
    # calculate overall precision
    precision = tp / (tp + fp)
    return precision


def weighted_precision(y_true, y_pred):
    """
    Function to calculate weighted precision

    Args:
        y_true: list of true values
        y_pred: list of predicted values
    """

    # find the number of classes
    num_classes = len(np.unique(y_true))

    # create a class:count directory
    class_counts = Counter(y_true)

    # initialize precision at 0
    precision = 0

    for class_ in range(num_classes):
        # all classes except the current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive for current class
        tp = true_positive(temp_true, temp_pred)

        # calculate false positive for current class
        fp = false_positive(temp_true, temp_pred)

        temp_precision = tp / (tp + fp)

        # multiply precision with count of samples in class
        weighted_precision = class_counts[class_] * temp_precision
        precision += weighted_precision
    
    # calculate average over all classes
    overall_precision = precision / len(y_true)
    return overall_precision


def weighted_f1(y_true, y_pred):
    """
    Function to calculate weighted f1

    Args:
        y_true: list of true values
        y_pred: list of predicted values
    """

    # find the number of classes
    num_classes = len(np.unique(y_true))

    # create a class:count directory
    class_counts = Counter(y_true)

    # initialize f1 at 0
    f1 = 0

    for class_ in range(num_classes):
        # all classes except the current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate precision and recall for current class
        p = precision(temp_true, temp_pred)
        r = recall(temp_true, temp_pred)

        # calculate f1 of class
        if p + r != 0:
            temp_f1 = 2 * p * r / (p + r)
        else:
            temp_f1 = 0

        # multiply f1 with count of samples in class
        weighted_f1 = class_counts[class_] * temp_f1
        f1 += weighted_f1
    
    # calculate average over all classes
    overall_f1 = f1 / len(y_true)
    return overall_f1