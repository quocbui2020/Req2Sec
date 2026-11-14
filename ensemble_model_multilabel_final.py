
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    classification_report, accuracy_score, hamming_loss,
    jaccard_score, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# === Load processed data ===
with open('processed_data_all.pkl', 'rb') as f:
    processed_data = pickle.load(f)

X_train, y_train = processed_data['train_resampled_title']  # tuple: (X_train, LP IDs)
X_test, y_test = processed_data['test_tfidf_title']         # y_test = LP IDs

# === Load Label Powerset mapping ===
mapping_df = pd.read_excel('label_powerset_filtered_for_smote_with_updated_data.xlsx')

def parse_cwe_labels(label_str):
    return eval(label_str) if isinstance(label_str, str) else []

label_powerset_mapping = mapping_df[['Label_Powerset_ID', 'CWE_Label']].drop_duplicates()
id_to_labels = dict(zip(label_powerset_mapping['Label_Powerset_ID'],
                        label_powerset_mapping['CWE_Label'].map(parse_cwe_labels)))

def map_ids_to_labels(ids, id_to_labels_map):
    return [id_to_labels_map.get(i, []) for i in ids]

# === Map LP IDs to CWE Labels ===
y_train_cwe = map_ids_to_labels(y_train, id_to_labels)
y_test_cwe = map_ids_to_labels(y_test, id_to_labels)

# === Binarize multilabel data ===
mlb = MultiLabelBinarizer()
Y_train_bin = mlb.fit_transform(y_train_cwe)
Y_test_bin = mlb.transform(y_test_cwe)

# === Define classifiers ===
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
clf_lr = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
clf_dt = DecisionTreeClassifier(random_state=42)
clf_knn = KNeighborsClassifier(n_neighbors=5)

# === Ensembles ===
soft_voting = VotingClassifier(
    estimators=[('rf', clf_rf), ('ann', clf_ann), ('lr', clf_lr)],
    voting='soft'
)

weighted_soft_voting = VotingClassifier(
    estimators=[('rf', clf_rf), ('ann', clf_ann), ('lr', clf_lr)],
    voting='soft',
    weights=[2, 1, 1]
)

hard_voting = VotingClassifier(
    estimators=[('rf', clf_rf), ('dt', clf_dt), ('knn', clf_knn)],
    voting='hard'
)

stacking = StackingClassifier(
    estimators=[('rf', clf_rf), ('ann', clf_ann), ('lr', clf_lr)],
    final_estimator=LogisticRegression(max_iter=1000),
    passthrough=False,
    cv=5
)

# === Train Models ===
soft_voting.fit(X_train, y_train)
weighted_soft_voting.fit(X_train, y_train)
hard_voting.fit(X_train, y_train)
stacking.fit(X_train, y_train)

# === Evaluation Function ===
def evaluate(model, X_test, y_test_lp, y_test_cwe, name):
    y_pred_lp = model.predict(X_test)
    y_pred_cwe = map_ids_to_labels(y_pred_lp, id_to_labels)
    Y_pred_bin = mlb.transform(y_pred_cwe)

    acc = accuracy_score(y_test_lp, y_pred_lp)
    h_loss = hamming_loss(Y_test_bin, Y_pred_bin)
    j_score = jaccard_score(Y_test_bin, Y_pred_bin, average='macro', zero_division=0)

    try:
        y_proba = model.predict_proba(X_test)
        if isinstance(y_proba, list):
            y_proba = np.mean(np.array([p for p in y_proba]), axis=0)
        roc = roc_auc_score(Y_test_bin, y_proba, average='macro', multi_class='ovr')
    except Exception:
        roc = None

    print(f"\n=== {name} ===")
    print(f"Accuracy (LP IDs): {acc:.4f}")
    print(f"Hamming Loss: {h_loss:.4f}, Jaccard Score: {j_score:.4f}, ROC-AUC: {roc}")

    report = classification_report(Y_test_bin, Y_pred_bin, target_names=mlb.classes_, zero_division=0, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(f"report_{name}.csv")

# === Evaluate All Ensembles ===
evaluate(soft_voting, X_test, y_test, y_test_cwe, "Soft_Voting_ML")
evaluate(weighted_soft_voting, X_test, y_test, y_test_cwe, "Weighted_Soft_Voting_ML")
evaluate(hard_voting, X_test, y_test, y_test_cwe, "Hard_Voting_ML")
evaluate(stacking, X_test, y_test, y_test_cwe, "Stacking_ML")
