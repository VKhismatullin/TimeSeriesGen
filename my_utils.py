import numpy as np
import scipy
from scipy.interpolate import CubicSpline
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


####
# 1. Some data transformations
###

class Jittering(object):
    """
    Apply Jiterring to a single time-series
    """
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sample):

        myNoise = np.random.normal(loc=0, scale=self.sigma, size=sample.shape)
        return sample + myNoise


class NormJittering:
    """
    Apply Jittering to a single time series that is normalized 
        by the standard deviation along each channel.
    """
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sample):
        noise = np.random.normal(
            loc=0, scale=self.sigma, size=sample.shape
        ) * np.std(sample, axis=0).reshape(1, -1)
        return sample + noise


class AdjustedNormTimeWarp:
    """
    Apply time warping using a cubic spline.
    """
    def __init__(self, sigma=0.2, knot=4):
        self.sigma = sigma
        self.knot = knot

    def __call__(self, sample):
        from scipy.interpolate import CubicSpline

        x = sample.copy().T[np.newaxis, :, :]
        orig_steps = np.arange(x.shape[1])

        random_warps = np.random.normal(
            loc=1.0, scale=self.sigma, size=(x.shape[0], self.knot + 2)
        )
        warp_steps = (
            np.ones((x.shape[2], 1))
            * np.linspace(0, x.shape[1] - 1.0, num=self.knot + 2)
        ).T

        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                time_warp = CubicSpline(
                    warp_steps[:, dim],
                    warp_steps[:, dim] * random_warps[i, :]
                )(orig_steps)
                scale = (x.shape[1] - 1) / time_warp[-1]
                ret[i, :, dim] = np.interp(
                    orig_steps,
                    np.clip(scale * time_warp, 0, x.shape[1] - 1),
                    pat[:, dim]
                ).T
        return ret.reshape(ret.shape[1], ret.shape[2]).T


def basic_series_transform(transform, prob=0.0):
    """
    Wrapper that applies a transform with given probability.
    Useful for data augmentation pipelines.
    """
    def wrapper(obj, *args, **kwargs):
        if np.random.binomial(1, prob):
            obj['series'] = transform(obj['series'], *args, **kwargs)
        return obj
    return wrapper


def autoenc_transform():
    """
    Wrapper that returns a tuple of (series, labels).
    Designed for autoencoder data pipelines.
    """
    def wrapper(obj, *args, **kwargs):
        return (obj['series'], obj['labels'])
    return wrapper


class SimpleDataset(Dataset):
    """
    Dataset class for time series data.
    Supports optional data augmentation.
    """
    def __init__(self, X, y, transforms=None):
        """
        Initialize dataset with features and labels.
        Optionally apply transformations to data.
        """
        super().__init__()
        self.transforms = transforms
        self.X = np.array(X)
        self.y = y
        self.counter = 0

    def __getitem__(self, idx):
        """
        Retrieve a single data item by index.
        Applies transformations if available.
        """
        obj = {'series': self.X[idx], 'labels': self.y[idx]}
        if self.transforms is not None:
            for transform in self.transforms:
                obj = transform(obj)
        return obj

    def __len__(self):
        """
        Return the total number of data items in the dataset.
        """
        return self.X.shape[0]

    
####
# 2. Importance Scores and DCI Metrics
####

import pandas as pd
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier


def get_imp(X, y):
    """
    Computes feature importances and accuracy from binary classification tasks
    using a Random Forest classifier for each column in y.

    Args:
        X (ndarray): Feature matrix of shape (n_samples, n_features).
        y (ndarray): Label matrix of shape (n_samples, 2), each column is a binary task.

    Returns:
        imp (ndarray): Feature importances of shape (n_features, 2).
        acc (float): Average classification accuracy across both tasks.
    """
    imp = np.zeros((X.shape[1], 2))
    acc = 0
    for class1 in range(2):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y[:, class1], stratify=y[:, class1], random_state=42)
        
        forest = RandomForestClassifier(random_state=0)
        forest.fit(X_train, y_train)
        acc += (forest.predict(X_test) == y_test).mean()

        importances = forest.feature_importances_
        imp[:, class1] = importances  # Collect feature importances
    return imp, acc / 2  # Average accuracy over both tasks


def get_DCI_score(impa):
    """
    Computes Disentanglement and Completeness scores based on feature importances.

    Args:
        impa (ndarray): Feature importance matrix of shape (n_features, n_factors).

    Returns:
        res_D (float): Disentanglement score (weighted by importance).
        res_C (float): Completeness score (average across factors).
    """
    impa[impa < 1e-10] = 1e-10  # Avoid log(0)
    P = impa / impa.sum(axis=1, keepdims=True)  # Normalize importances
    rho = impa.mean(axis=1)  # Importance mean per feature

    H = -np.sum(P * np.log(P), axis=1)  # Entropy per feature
    max_H = np.log(P.shape[1])
    D = 1 - H / max_H  # Disentanglement
    res_D = np.sum(rho * D)

    H = -np.sum(impa * np.log(impa), axis=0)  # Completeness entropy
    max_H = np.log(P.shape[0])
    C = 1 - H / max_H  # Completeness
    res_C = np.mean(C)
    return res_D, res_C


def plot_relevance(new, y, title=None):
    """
    Trains a Random Forest classifier and plots feature importances with error bars.

    Args:
        new (ndarray): Feature matrix of shape (n_samples, n_features).
        y (ndarray): Target labels.
        title (str, optional): Custom plot title.

    Displays:
        Bar chart of mean decrease in impurity (MDI) with standard deviation.
    """
    import matplotlib.pyplot as plt
    X_train, X_test, y_train, y_test = train_test_split(new, y, 
                        stratify=y, random_state=42)

    feature_names = [f"feature {i}" for i in range(X_train.shape[1])]
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_train, y_train)
    
    print('ACC:', np.mean(forest.predict(X_train) == y_train))

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    forest_importances = pd.Series(importances, index=feature_names)
    
    fig, ax = plt.subplots(figsize=(14,6))
    forest_importances.plot.bar(yerr=std, ax=ax)

    ax.set_title("Feature importances using MDI" if title is None else title)
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    