import numpy as np

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, recall_score, precision_score

def print_scores(y_true, y_pred):
    print(f"F1: {f1_score(y_true, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"Precision: {precision_score(y_true, y_pred)}")
    print(f"ROC AUC: {roc_auc_score(y_true, y_pred)}")
    print(f"Recall: {recall_score(y_true, y_pred)}")

def get_best_thr(tpr, fpr, thrs):

    # Find the best threshold (maximize Youden’s J statistic)
    j_scores = tpr - fpr  # Compute J = TPR - FPR
    best_index = np.argmax(j_scores)  # Index of best threshold
    return thrs[best_index]