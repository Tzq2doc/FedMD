import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.datasets import cifar10, cifar100, mnist
import scipy.io as sio


def load_MNIST_data(standarized = False, verbose = False):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    if standarized: 
        X_train = X_train/255
        X_test = X_test/255
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_test -= mean_image
    
    if verbose == True: 
        print("MNIST dataset ... ")
        print("X_train shape :", X_train.shape)
        print("X_test shape :", X_test.shape)
        print("y_train shape :", y_train.shape)
        print("y_test shape :", y_test.shape)
    
    return X_train, y_train, X_test, y_test


def load_EMNIST_data(file, verbose = False, standarized = False):
    """
    file should be the downloaded EMNIST file in .mat format.
    """    
    mat = sio.loadmat(file)
    data = mat["dataset"]
    
    X_train = data['train'][0,0]['images'][0,0]
    X_train = X_train.reshape((X_train.shape[0], 28, 28), order = "F")
    y_train = data['train'][0,0]['labels'][0,0]
    y_train = np.squeeze(y_train)
    y_train -= 1 #y_train is zero-based
    
    X_test = data['test'][0,0]['images'][0,0]
    X_test= X_test.reshape((X_test.shape[0], 28, 28), order = "F")
    y_test = data['test'][0,0]['labels'][0,0]
    y_test = np.squeeze(y_test)
    y_test -= 1 #y_test is zero-based
    
    if standarized: 
        X_train = X_train/255
        X_test = X_test/255
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_test -= mean_image
    

    if verbose == True: 
        print("EMNIST-letter dataset ... ")
        print("X_train shape :", X_train.shape)
        print("X_test shape :", X_test.shape)
        print("y_train shape :", y_train.shape)
        print("y_test shape :", y_test.shape)
    
    return X_train, y_train, X_test, y_test


def generate_partial_data(X, y, class_in_use = "all", verbose = False):
    if class_in_use == "all":
        idx = np.ones_like(y, dtype = bool)
    else:
        idx = [y == i for i in class_in_use]
        idx = np.any(idx, axis = 0)
    X_incomplete, y_incomplete = X[idx], y[idx]
    if verbose == True:
        print("X shape :", X_incomplete.shape)
        print("y shape :", y_incomplete.shape)
    return X_incomplete, y_incomplete



def generate_bal_private_data(X, y, N_parties = 10, classes_in_use = range(11), 
                              N_samples_per_class = 20, data_overlap = False):
    """
    Input: 
    -- N_parties : int, number of collaboraters in this activity;
    -- classes_in_use: array or generator, the classes of EMNIST-letters dataset 
    (0 <= y <= 25) to be used as private data; 
    -- N_sample_per_class: int, the number of private data points of each class for each party
    
    return: 
    
    """
    priv_data = [None] * N_parties
    combined_idx = np.array([], dtype = np.int16)
    for cls in classes_in_use:
        idx = np.where(y == cls)[0]
        idx = np.random.choice(idx, N_samples_per_class * N_parties, 
                               replace = data_overlap)
        combined_idx = np.r_[combined_idx, idx]
        for i in range(N_parties):           
            idx_tmp = idx[i * N_samples_per_class : (i + 1)*N_samples_per_class]
            if priv_data[i] is None:
                tmp = {}
                tmp["X"] = X[idx_tmp]
                tmp["y"] = y[idx_tmp]
                tmp["idx"] = idx_tmp
                priv_data[i] = tmp
            else:
                priv_data[i]['idx'] = np.r_[priv_data[i]["idx"], idx_tmp]
                priv_data[i]["X"] = np.vstack([priv_data[i]["X"], X[idx_tmp]])
                priv_data[i]["y"] = np.r_[priv_data[i]["y"], y[idx_tmp]]
                
                
    total_priv_data = {}
    total_priv_data["idx"] = combined_idx
    total_priv_data["X"] = X[combined_idx]
    total_priv_data["y"] = y[combined_idx]
    return priv_data, total_priv_data


def generate_alignment_data(X, y, N_alignment = 3000):
    
    split = StratifiedShuffleSplit(n_splits=1, train_size= N_alignment)
    if N_alignment == "all":
        alignment_data = {}
        alignment_data["idx"] = np.arange(y.shape[0])
        alignment_data["X"] = X
        alignment_data["y"] = y
        return alignment_data
    for train_index, _ in split.split(X, y):
        X_alignment = X[train_index]
        y_alignment = y[train_index]
    alignment_data = {}
    alignment_data["idx"] = train_index
    alignment_data["X"] = X_alignment
    alignment_data["y"] = y_alignment
    
    return alignment_data