import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, hamming_loss,
    jaccard_score, roc_auc_score
)

# === Load processed data ===
with open('processed_data_all.pkl', 'rb') as f:
    processed_data = pickle.load(f)

X_train, y_train = processed_data['train_resampled_combined']
X_test, y_test = processed_data['test_tfidf_combined']

# === Load LP to CWE mapping ===
mapping_df = pd.read_excel('label_powerset_filtered_for_smote_with_updated_data.xlsx')

def parse_cwe_labels(label_str):
    return eval(label_str) if isinstance(label_str, str) else []

label_map = dict(zip(mapping_df['Label_Powerset_ID'], mapping_df['CWE_Label'].map(parse_cwe_labels)))

def map_ids_to_labels(ids, id_map): return [id_map.get(i, []) for i in ids]

y_train_cwe = map_ids_to_labels(y_train, label_map)
y_test_cwe = map_ids_to_labels(y_test, label_map)

# === Binarization ===
mlb = MultiLabelBinarizer()
Y_train_bin = mlb.fit_transform(y_train_cwe)
Y_test_bin = mlb.transform(y_test_cwe)

# === Models ===
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
clf_lr = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
clf_knn = KNeighborsClassifier(n_neighbors=5)

# === Ensemble Definitions ===
ensembles = {
    "Soft_Voting_TD": VotingClassifier(
        estimators=[('rf', clf_rf), ('ann', clf_ann), ('lr', clf_lr)],
        voting='soft'),
    
    "Weighted_Soft_Voting_TD": VotingClassifier(
        estimators=[('rf', clf_rf), ('ann', clf_ann), ('lr', clf_lr)],
        voting='soft', weights=[2, 1, 1]),
    
    "Hard_Voting_TD": VotingClassifier(
        estimators=[('rf', clf_rf), ('ann', clf_ann), ('knn', clf_knn)],
        voting='hard'),
    
    "Stacking_TD": StackingClassifier(
        estimators=[('rf', clf_rf), ('ann', clf_ann), ('lr', clf_lr)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5)
}

# === Evaluate + Save ===
def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred_lp = model.predict(X_test)
    y_pred_cwe = map_ids_to_labels(y_pred_lp, label_map)
    Y_pred_bin = mlb.transform(y_pred_cwe)

    # Main metrics
    acc = accuracy_score(y_test, y_pred_lp)
    h_loss = hamming_loss(Y_test_bin, Y_pred_bin)
    j_macro = jaccard_score(Y_test_bin, Y_pred_bin, average='macro', zero_division=0)
    j_micro = jaccard_score(Y_test_bin, Y_pred_bin, average='micro', zero_division=0)
    try:
        y_proba = model.predict_proba(X_test)
        if isinstance(y_proba, list):
            y_proba = np.mean(np.array([p for p in y_proba]), axis=0)
        roc_auc = roc_auc_score(Y_test_bin, y_proba, average='macro', multi_class='ovr')
    except Exception:
        roc_auc = None

    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {acc:.4f}, Hamming Loss: {h_loss:.4f}, Jaccard (macro): {j_macro:.4f}, ROC-AUC: {roc_auc}")

    # === Save classification report ===
    report_dict = classification_report(Y_test_bin, Y_pred_bin, target_names=mlb.classes_, output_dict=True, zero_division=0)
    pd.DataFrame(report_dict).transpose().to_csv(f"report_{model_name}.csv")

    # === Save confusion matrix ===
    cm = confusion_matrix(y_test, y_pred_lp)
    pd.DataFrame(cm).to_csv(f"confusion_matrix_{model_name}.csv", index=False)

    # === Save summary metrics ===
    summary = pd.DataFrame([{
        "Model": model_name,
        "Accuracy": acc,
        "Hamming Loss": h_loss,
        "Jaccard Macro": j_macro,
        "Jaccard Micro": j_micro,
        "ROC AUC": roc_auc
    }])
    summary.to_csv(f"summary_metrics_{model_name}.csv", index=False)

# === Run for all ensemble models ===
for name, model in ensembles.items():
    evaluate_model(model, name)
