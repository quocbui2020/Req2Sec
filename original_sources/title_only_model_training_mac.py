import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    hamming_loss, accuracy_score, jaccard_score, label_ranking_average_precision_score,
    coverage_error, recall_score
)

# === Config: use LinearSVC for speed in SVM ===
USE_LINEAR_SVC = True  # Set to False to use standard SVC with probability=True

# === Setup local save path ===
save_path = "/Users/rohan/Research/Vulnerability Mapping/title_only_model_training_balanced_final"
os.makedirs(save_path, exist_ok=True)

# === Load data ===
with open('processed_data_all.pkl', 'rb') as f:
    processed_data = pickle.load(f)

X_train, y_train = processed_data['train_resampled_title']
X_test, y_test = processed_data['test_tfidf_title']

# === Load Label Powerset mapping ===
mapping_df = pd.read_excel('label_powerset_filtered_for_smote_with_updated_data.xlsx')

def parse_cwe_labels(label_str):
    return eval(label_str) if isinstance(label_str, str) else []

label_powerset_mapping = mapping_df[['Label_Powerset_ID', 'CWE_Label']].drop_duplicates()
id_to_labels = dict(zip(label_powerset_mapping['Label_Powerset_ID'],
                        label_powerset_mapping['CWE_Label'].map(parse_cwe_labels)))

def map_ids_to_labels(ids, id_to_labels_map):
    return [id_to_labels_map.get(i, []) for i in ids]

# === Define models ===
model_names_to_run = [
    'Decision Tree',
    'Random Forest',
    'Naive Bayes',
    'Artificial Neural Network (ANN)',
    'Support Vector Machine (SVM)',
    'Logistic Regression (LR)',
    'K-Nearest Neighbors (KNN)'
]

models_dict = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'Artificial Neural Network (ANN)': MLPClassifier(random_state=42),
    'Support Vector Machine (SVM)': LinearSVC(random_state=42, max_iter=100000) if USE_LINEAR_SVC else SVC(random_state=42, probability=True),
    'Logistic Regression (LR)': LogisticRegression(random_state=42, max_iter=1000),
    'K-Nearest Neighbors (KNN)': KNeighborsClassifier()
}

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# === Helper for Balanced Accuracy ===
def compute_balanced_accuracy_multilabel(y_true, y_pred):
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    return np.mean(recalls)

for name in model_names_to_run:
    print(f"\n=== Training: {name} ===")
    model = models_dict[name]

    if name == 'Naive Bayes':
        X_tr = X_train.toarray()
        X_te = X_test.toarray()
    else:
        X_tr = X_train
        X_te = X_test

    if name in ['Logistic Regression (LR)', 'Support Vector Machine (SVM)', 'Artificial Neural Network (ANN)']:
        scaler = StandardScaler(with_mean=False) if hasattr(X_tr, 'toarray') else StandardScaler()
        X_tr = scaler.fit_transform(X_tr if not hasattr(X_tr, 'toarray') else X_tr.toarray())
        X_te = scaler.transform(X_te if not hasattr(X_te, 'toarray') else X_te.toarray())

    # Cross-validation
    cv_scores = cross_val_score(model, X_tr, y_train, cv=kf, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean Accuracy: {np.mean(cv_scores):.4f}, Std: {np.std(cv_scores):.4f}")

    # Train model
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_pred_prob = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None

    # Map LP IDs to multilabels
    y_test_labels = map_ids_to_labels(y_test, id_to_labels)
    y_pred_labels = map_ids_to_labels(y_pred, id_to_labels)

    mlb = MultiLabelBinarizer()
    y_test_bin = mlb.fit_transform(y_test_labels)
    y_pred_bin = mlb.transform(y_pred_labels)

    # Classification report
    report = classification_report(y_test_bin, y_pred_bin, target_names=mlb.classes_, zero_division=0)
    print(report)
    with open(f"{save_path}/report_{name}.txt", "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test_bin.argmax(axis=1), y_pred_bin.argmax(axis=1))
    cm_df = pd.DataFrame(cm, index=mlb.classes_, columns=mlb.classes_)
    cm_df.to_csv(f"{save_path}/conf_matrix_{name}.csv")

    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=mlb.classes_, yticklabels=mlb.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix: {name}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{save_path}/conf_matrix_{name}.png")
    plt.close()

    # Filter out "Clean" class
    try:
        clean_index = list(mlb.classes_).index("Clean")
        keep_indices = [i for i in range(len(mlb.classes_)) if i != clean_index]
        y_test_bin_no_clean = y_test_bin[:, keep_indices]
        y_pred_bin_no_clean = y_pred_bin[:, keep_indices]
    except ValueError:
        y_test_bin_no_clean = y_test_bin
        y_pred_bin_no_clean = y_pred_bin

    # Metrics summary
    def compute_metrics(y_true, y_pred, note=""):
        metrics = {
            f'{note}Subset Accuracy': accuracy_score(y_true, y_pred),
            f'{note}Balanced Accuracy': compute_balanced_accuracy_multilabel(y_true, y_pred),
            f'{note}Hamming Loss': hamming_loss(y_true, y_pred),
            f'{note}Jaccard Score (macro)': jaccard_score(y_true, y_pred, average='macro')
        }
        return metrics

    metrics_with_clean = compute_metrics(y_test_bin, y_pred_bin, "with_clean_")
    metrics_without_clean = compute_metrics(y_test_bin_no_clean, y_pred_bin_no_clean, "no_clean_")

    pd.Series({**metrics_with_clean, **metrics_without_clean}).to_csv(f"{save_path}/metrics_{name}.csv")
    print(f"Metrics saved for {name}")

    # Clean up memory
    del model, y_pred, y_pred_prob, X_tr, X_te
    gc.collect()
