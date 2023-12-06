import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

import torch

def split_train_test_val(n_samples, test_samples=0.1, val_samples=0.1, seed=0):
    """
    Returns masks to split the graph into train / test / validation sets.

    Parameters
    ----------
    n_samples : int
        Number of samples
    test_ratio : float or int
        Ratio of test samples or number of test samples
    val_ratio : float or int
        Ratio of validation samples or number of validation samples

    Returns
    -------
    train: np.ndarray
        Boolean mask
    test: np.ndarray
        Boolean mask
    validation: np.ndarray
        Boolean mask
    """
    if seed:
        np.random.seed(seed)
    
    if(test_samples < 1):
        test_samples = int(np.ceil(n_samples * test_samples))
    if(val_samples < 1):
        val_samples = int(np.ceil(n_samples * val_samples))
    
    # test
    index = np.random.choice(n_samples, test_samples, replace=False)
    test = np.zeros(n_samples, dtype=bool)
    test[index] = 1

    # validation
    index = np.random.choice(np.argwhere(~test).ravel(), val_samples, replace=False)
    val = np.zeros(n_samples, dtype=bool)
    val[index] = 1

    # train
    train = np.ones(n_samples, dtype=bool)
    train[test] = 0
    train[val] = 0
    return train, test, val

def eval_model(model, graph, features, labels, mask):
    '''Evaluate the model in terms of accuracy.'''
    model.eval()
    with torch.no_grad():
        output = model(graph, features)
        labels_pred = torch.max(output, dim=1)[1]
        score = np.mean(np.array(labels[mask]) == np.array(labels_pred[mask]))
    return score

def plot_loss(loss):
    '''Plot loss.'''
    plt.figure( figsize=(6,6) )
    plt.plot(loss, label='train loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_accuracy(train_score, val_score):
    '''Plot train and validation accuracy.'''
    plt.figure( figsize=(6,6) )
    plt.plot(train_score, label='train accuracy')
    plt.plot(val_score, label='validation accuracy')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.show()