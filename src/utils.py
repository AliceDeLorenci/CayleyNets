import numpy as np

import matplotlib.pyplot as plt
# plt.rcParams["font.family"] = "Times New Roman"

import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import time
import datetime

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

def train(model, optimizer, edge_index, features, labels, train_mask, val_mask, epochs, verbose=True):
    '''Train model.'''

    # save loss values for plotting
    loss_values = []
    val_score = []
    train_score = []

    for e in range(epochs):
        start = time.time()

        # Compute output
        logp = model(features, edge_index)

        # Compute loss
        loss = F.nll_loss(logp[train_mask], labels[train_mask])
        loss_values.append(loss.item())

        # Perform backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluation
        score = eval_pytorch_geometric_model(model, edge_index, features, labels, train_mask)
        train_score.append(score)

        score = eval_pytorch_geometric_model(model, edge_index, features, labels, val_mask)
        val_score.append(score)

        # Print loss
        end = time.time()
        if verbose:
            print("Epoch {:02d} | Loss {:.3f} | Accuracy (validation) {:.3f} | Elapsed time: {:.2f}s".format(e, loss.item(), score, end - start))
        
    return loss_values, val_score, train_score

def eval_dgl_model(model, graph, features, labels, mask):
    '''Evaluate the model in terms of accuracy.'''
    model.eval() 

    labels_cpu = labels.cpu().detach()
    mask_cpu = mask.cpu().detach()
    with torch.no_grad():
        output = model(graph, features).cpu().detach()
        labels_pred = torch.max(output, dim=1)[1]
        score = np.mean(np.array(labels_cpu[mask_cpu]) == np.array(labels_pred[mask_cpu]))
    return score

def eval_pytorch_geometric_model(model, edge_index, features, labels, mask):
    '''Evaluate the model in terms of accuracy.'''
    model.eval()

    labels_cpu = labels.cpu().detach()
    mask_cpu = mask.cpu().detach()
    with torch.no_grad():
        output = model(features, edge_index).cpu().detach()
        labels_pred = torch.max(output, dim=1)[1]
        score = np.mean(np.array(labels_cpu[mask_cpu]) == np.array(labels_pred[mask_cpu]))
    return score

def plot_loss(loss):
    '''Plot loss.'''
    plt.figure( figsize=(4,4) )
    plt.plot(loss, label='train loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_accuracy(train_score, val_score):
    '''Plot train and validation accuracy.'''
    plt.figure( figsize=(4,4) )
    plt.plot(train_score, label='train accuracy')
    plt.plot(val_score, label='validation accuracy')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.show()

def save_results(results, model_name='', dataset_name=''):

    ts = datetime.datetime.now()
    ts = '{}-{}-{}-{}-{}-{}'.format(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)

    fname = './results/{}_{}_{}.pkl'.format(dataset_name, model_name, ts)
    with open(fname, 'wb') as f:
        pickle.dump(results, f)
            
    with open(fname, 'rb') as f:
        loaded_results = pickle.load(f)

    return loaded_results

def load_results(fname):
    with open(fname, 'rb') as f:
        loaded_results = pickle.load(f)
    return loaded_results