import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score, roc_curve
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from xgboost import XGBClassifier

def print_scores(y_true, y_pred):
    print(f"F1: {f1_score(y_true, y_pred):.3f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"ROC AUC: {roc_auc_score(y_true, y_pred):.3f}")
    print(f"Recall: {recall_score(y_true, y_pred):.3f}")

def get_best_thr(tpr, fpr, thrs):

    # Find the best threshold (maximize Youden’s J statistic)
    j_scores = tpr - fpr  # Compute J = TPR - FPR
    best_index = np.argmax(j_scores)  # Index of best threshold
    return thrs[best_index]

def split_data(X, y, test_size = 0.25, random_state = 42, shuffle = True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=random_state, shuffle=shuffle)

    return X_train, X_test, y_train, y_test

def plot_roc(roc_train, roc_test, roc_bm = None):

    fpr_train, tpr_train = roc_train
    fpr_test, tpr_test = roc_test

    if roc_bm is None:
        plt.plot(fpr_train, tpr_train, color = 'b', label = 'Train ROC')
        plt.plot(fpr_test, tpr_test, alpha = 0.75, color = 'r', label = 'Test ROC')
        plt.plot(fpr_train, fpr_train, color = 'y', label = 'Random Model ROC')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title("ROC Curve Train/Test/Random Model")
        plt.legend()
    else:
        fpr_bm, tpr_bm = roc_bm
        plt.plot(fpr_train, tpr_train, color = 'b', label = 'Train ROC')
        plt.plot(fpr_test, tpr_test, alpha = 0.75, color = 'r', label = 'Test ROC')
        plt.plot(fpr_bm, tpr_bm, color = 'y', label = 'Best Model ROC')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title("ROC Curve Train/Test/Best Model")
        plt.legend()

def train_and_predict_xgbc(X_train, y_train, X_test):

    xgbc = XGBClassifier()

    xgbc.fit(X_train, y_train)
    y_train_pred = xgbc.predict_proba(X_train)[:, 1]
    y_test_pred = xgbc.predict_proba(X_test)[:, 1]

    return y_train_pred, y_test_pred, xgbc

def confusion_matrix(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return pd.DataFrame({'Predicted Positive' : [tp, fp],
              'Predicted Negative' : [fn, tn]}, index = ['Measured Positive', 'Measured Negative'])

def calculate_roc(y_true, y_pred):

    y_train, y_test = y_true
    y_pred_train, y_pred_test = y_pred

    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train)
    fpr_test, tpr_test, thr = roc_curve(y_test, y_pred_test)

    return (fpr_train, tpr_train), (fpr_test, tpr_test), thr