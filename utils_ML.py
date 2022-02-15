# IMPORTS
import csv
from email import message
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import pathlib

from collections import Counter
from collections.abc import Iterable
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.cluster import KMeans
from typing import Collection


# FUNCS
# To get model scoring for regression models
def reg_scoring(name, model, x, y, set='test'):
    name = f'{name.upper()} ({set} data)'
    preds = model.predict(x)

    metrics = pd.DataFrame({name: [f'{model.score(x, y):.10f}',
                                   f'{mean_absolute_error(y, preds):.10f}',
                                   f'{mean_absolute_percentage_error(y, preds):.10f}',
                                   f'{mean_squared_error(y, preds):.10f}',
                                   f'{np.sqrt(mean_squared_error(y, preds)):.10f}']},
                           index=[['Score (R2 coef.)', 'MAE', 'MAPE', 'MSE', 'RMSE']])

    return metrics

# To get model scoring for classification models
def clas_scoring(name, model, x, y, set='test'):
    name = f'{name.upper()} ({set} data)'
    preds = model.predict(x)

    metrics = pd.DataFrame({name: [f'{accuracy_score(y, preds):.10f}',
                                   f'{precision_score(y, preds):.10f}',
                                   f'{recall_score(y, preds):.10f}',
                                   f'{f1_score(y, preds):.10f}',
                                   f'{roc_auc_score(y, preds):.10f}']},
                           index=[['Accuracy (TP + TN/TT)', 'Precision (TP/TP + FP)', 'Recall (TP/TP + FN)',
                                   'F1 (har_mean Ac, Re)', 'ROC AUC']])

    return metrics

# Show polynomial regression
def viz_poly(model, x, x_poly, y):
    plt.scatter(x, y, color='darkred')
    plt.plot(x, model.predict(x_poly), color='cornflowerblue')
    plt.title(nameof(model, globals()))
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()

    return

# To return the name of an object
def nameof(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]