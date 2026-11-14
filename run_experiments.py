"""Runnable baseline/ensemble workflow reconstructed from recovered research scripts.

Historical model choices are preserved where identifiable. Evaluation defects are
corrected; results are not guaranteed to equal published dissertation/article scores.
"""
import argparse
from pathlib import Path
import gc
import pandas as pd
from scipy import sparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from common import ROOT, load_data, evaluate

BASELINE_NAMES = ['dt', 'rf', 'nb', 'ann', 'svm', 'lr', 'knn']
ENSEMBLE_NAMES = ['hard', 'soft', 'weighted_soft', 'stacking']

def baseline_models():
    return {
        'dt': DecisionTreeClassifier(random_state=42),
        'rf': RandomForestClassifier(random_state=42),
        'nb': GaussianNB(),
        'ann': MLPClassifier(random_state=42),
        'svm': LinearSVC(random_state=42, max_iter=100000),
        'lr': LogisticRegression(random_state=42, max_iter=1000),
        'knn': KNeighborsClassifier(),
    }

def ensemble_model(name, config):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    lr = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    if name == 'hard':
        # Historical recovered scripts differ: title uses DT, combined uses ANN.
        third = DecisionTreeClassifier(random_state=42) if config == 'title' else ann
        return VotingClassifier(estimators=[('rf', rf), ('second', third), ('knn', knn)], voting='hard')
    estimators = [('rf', rf), ('ann', ann), ('lr', lr)]
    if name == 'soft':
        return VotingClassifier(estimators=estimators, voting='soft')
    if name == 'weighted_soft':
        return VotingClassifier(estimators=estimators, voting='soft', weights=[2, 1, 1])
    if name == 'stacking':
        return StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), cv=5)
    raise ValueError(name)

def transform_baseline(name, x_train, x_test):
    if name == 'nb':
        # GaussianNB does not accept sparse input; conversion can require many GB.
        return (x_train.toarray() if sparse.issparse(x_train) else x_train,
                x_test.toarray() if sparse.issparse(x_test) else x_test)
    if name in ('ann', 'svm', 'lr'):
        # Sparse-preserving scaling avoids original script's eager densification.
        scaler = StandardScaler(with_mean=False)
        return scaler.fit_transform(x_train), scaler.transform(x_test)
    return x_train, x_test

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', choices=['title', 'combined', 'both'], default='both')
    p.add_argument('--family', choices=['baselines', 'ensembles', 'both'], default='both')
    p.add_argument('--models', nargs='*', help='Subset: dt rf nb ann svm lr knn hard soft weighted_soft stacking')
    p.add_argument('--cv', action='store_true', help='Run historical-style 5-fold CV on already resampled training data (leakage risk)')
    p.add_argument('--output', type=Path, default=ROOT / 'results')
    args = p.parse_args()
    configs = ['title', 'combined'] if args.config == 'both' else [args.config]
    allowed = (BASELINE_NAMES if args.family in ('baselines', 'both') else []) + (ENSEMBLE_NAMES if args.family in ('ensembles', 'both') else [])
    names = args.models if args.models is not None else allowed
    invalid = set(names) - set(allowed)
    if invalid:
        p.error(f'Invalid model(s) for family: {sorted(invalid)}')
    for config in configs:
        print(f'Loading {config} dataset...', flush=True)
        x_train, y_train, x_test, y_test, mapping, mlb, y_true = load_data(config)
        print(f'Train {x_train.shape}; test {x_test.shape}; CWE labels {len(mlb.classes_)}', flush=True)
        results = []
        for name in names:
            print(f'Training {config}/{name}...', flush=True)
            if name in BASELINE_NAMES:
                model = baseline_models()[name]
                train, test = transform_baseline(name, x_train, x_test)
            else:
                model = ensemble_model(name, config)
                train, test = x_train, x_test
            if args.cv and name in BASELINE_NAMES:
                counts = pd.Series(y_train).value_counts()
                if counts.min() < 5:
                    print('CV skipped: rare LP class has fewer than five training samples')
                else:
                    cv = StratifiedKFold(5, shuffle=True, random_state=42)
                    scores = cross_val_score(model, train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
                    print(f'CV accuracy mean={scores.mean():.5f}, std={scores.std():.5f}')
            model.fit(train, y_train)
            result = evaluate(name, model, test, y_test, mapping, mlb, y_true, args.output / config)
            print(f"{config}/{name}: micro-F1={result['micro_f1']:.4f} macro-F1={result['macro_f1']:.4f}", flush=True)
            results.append(result)
            del model, train, test
            gc.collect()
        out = args.output / config
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out / 'summary.csv', index=False)

if __name__ == '__main__':
    main()
