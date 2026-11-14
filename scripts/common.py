"""Shared I/O and multilabel evaluation for the archived LP experiments."""
import ast
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, hamming_loss, jaccard_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]

def parse_labels(value):
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return [x.strip() for x in str(value).split('|') if x.strip()]
    if isinstance(parsed, (list, tuple, set)):
        return [str(v) for v in parsed]
    return [str(parsed)]

def load_data(config):
    # WARNING: pickle is executable; only load a trusted, unmodified artifact.
    with (ROOT / 'data' / 'processed_data_all.pkl').open('rb') as f:
        data = pickle.load(f)
    suffix = 'title' if config == 'title' else 'combined'
    x_train, y_train = data[f'train_resampled_{suffix}']
    x_test, y_test = data[f'test_tfidf_{suffix}']
    mapping = pd.read_excel(ROOT / 'data' / 'label_powerset_filtered_for_smote_with_updated_data.xlsx')
    required = {'Label_Powerset_ID', 'CWE_Label'}
    if not required.issubset(mapping):
        raise ValueError(f'Missing mapping columns: {required - set(mapping)}')
    label_map = {}
    for lp_id, raw in zip(mapping['Label_Powerset_ID'], mapping['CWE_Label']):
        labels = tuple(sorted(set(parse_labels(raw))))
        lp_id = int(lp_id)
        if lp_id in label_map and label_map[lp_id] != labels:
            raise ValueError(f'Inconsistent CWE mapping for LP ID {lp_id}')
        label_map[lp_id] = labels
    missing = set(np.unique(np.r_[y_train, y_test])) - set(label_map)
    if missing:
        raise ValueError(f'Missing Label Powerset IDs: {sorted(missing)[:20]}')
    mlb = MultiLabelBinarizer(classes=sorted({label for labels in label_map.values() for label in labels}))
    mlb.fit([[]])
    y_true_bin = mlb.transform([label_map[int(i)] for i in y_test])
    return x_train, y_train, x_test, y_test, label_map, mlb, y_true_bin

def evaluate(name, model, x_test, y_test, label_map, mlb, y_true_bin, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(x_test)
    unknown = set(np.unique(y_pred)) - set(label_map)
    if unknown:
        raise ValueError(f'Unknown predicted LP IDs: {unknown}')
    y_pred_bin = mlb.transform([label_map[int(i)] for i in y_pred])
    report = classification_report(y_true_bin, y_pred_bin, target_names=[str(c) for c in mlb.classes_], output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out_dir / f'{name}_classification_report.csv')
    lp_classes = sorted(set(np.unique(y_test)) | set(np.unique(y_pred)))
    cm = confusion_matrix(y_test, y_pred, labels=lp_classes)
    pd.DataFrame(cm, index=lp_classes, columns=lp_classes).to_csv(out_dir / f'{name}_lp_confusion_matrix.csv')
    summary = {
        'model': name,
        'micro_f1': report['micro avg']['f1-score'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'micro_precision': report['micro avg']['precision'],
        'micro_recall': report['micro avg']['recall'],
        'subset_accuracy': accuracy_score(y_true_bin, y_pred_bin),
        'lp_accuracy': accuracy_score(y_test, y_pred),
        'hamming_loss': hamming_loss(y_true_bin, y_pred_bin),
        'macro_jaccard': jaccard_score(y_true_bin, y_pred_bin, average='macro', zero_division=0),
    }
    return summary
