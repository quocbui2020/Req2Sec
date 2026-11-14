# Requirement Vulnerability Prediction — ML Replication Artifacts

Machine-learning replication materials accompanying **“From Requirements to Security Insights: Predicting Vulnerability Classes for Software Requirements via Function-Level Traceability”** (REFSQ 2027 manuscript).

## Scope and provenance

This package reproduces the **model-training/evaluation stage** starting from the recovered, preprocessed data. It does **not** reproduce Bugzilla/NVD extraction, function-level mining, R-to-SC/V-to-SC joins, text cleaning, train/test partitioning, TF-IDF fitting, or SMOTE+ENN resampling. Those steps are **already represented in the supplied pickle**. It does not include the original full R-to-V mapping.

- `data/processed_data_all.pkl` — **original recovered** Python pickle with resampled training matrices and untouched test matrices for title-only and title+description experiments; fitted TF-IDF vectorizers also present.
- `data/label_powerset_filtered_for_smote_with_updated_data.xlsx` — **original recovered** requirement-level label powerset mapping, including CWE label sets.
- `original_sources/` — **unaltered recovered** historical baseline and ensemble source files, plus one clearly marked **reconstructed** combined-baseline script. These historical scripts have known portability/evaluation issues and are supplied for provenance, **not** as the recommended entry point.
- `scripts/run_experiments.py`, `scripts/common.py`, `scripts/check_dataset.py` — **newly reconstructed and corrected** portable workflow. Not claimed to be the exact historical implementation that produced the paper's results.

**Do not claim exact numerical replication until a full run is compared with the article's Table 3.** Corrections to metrics and handling of sparse data may change results; model versions/hyperparameters may also differ from the original experiments.

## Requirements

Python 3.10–3.12 recommended; scikit-learn 1.6.1 matches the version identified in the recovered pickle. Other dependencies are listed in `requirements.txt` without historically verified exact pins. Large resampled matrices (approximately 361k title-only / 389k combined training rows) and some estimators may require substantial RAM and CPU time. A complete run may take hours or longer. GaussianNB densifies the training matrix and may require multiple GB of RAM.

**Security:** Python pickle loading can execute arbitrary code. Only load this pickle if you trust its provenance; do not accept untrusted replacement pickle files.

## Setup

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/check_dataset.py
```

## Run

Run all seven baselines and four ensembles for both input configurations:

```bash
python scripts/run_experiments.py
```

Or run a single configuration/family, or a small smoke test:

```bash
python scripts/run_experiments.py --config title --family baselines
python scripts/run_experiments.py --config title --family ensembles
python scripts/run_experiments.py --config combined --family baselines
python scripts/run_experiments.py --config combined --family ensembles
python scripts/run_experiments.py --config title --family baselines --models dt
```

Models: `dt`, `rf`, `nb`, `ann`, `svm`, `lr`, `knn`, `hard`, `soft`, `weighted_soft`, `stacking`.

Optional `--cv` runs five-fold stratified cross-validation on the **already resampled** training data, reflecting the archived script's approach. **Caution:** this can give optimistic estimates because resampling was performed before CV; it is not leakage-free hyperparameter tuning. Default runs train on the supplied training partition and evaluate on the held-out test partition, without cross-validation.

## Outputs

Results are written to `results/title/` and `results/combined/`:

- `summary.csv` with micro/macro/weighted F1, precision/recall, exact-set subset accuracy, LP-ID accuracy, Hamming loss and Jaccard score.
- `<model>_classification_report.csv` — per-CWE multilabel classification report.
- `<model>_lp_confusion_matrix.csv` — multiclass **Label Powerset** confusion matrix (not a CWE multilabel confusion matrix).

The shared evaluation uses a fixed CWE label vocabulary from the mapping workbook, `ast.literal_eval` instead of `eval`, and checks that LP IDs resolve consistently. ROC-AUC is intentionally omitted because the historical scripts compare LP-class probabilities against CWE-level targets, which are incompatible.

## Model configurations

Seven baselines: decision tree, random forest, Gaussian Naive Bayes, MLP/ANN, linear SVM, logistic regression, and KNN. Ensembles: hard voting, soft voting, weighted soft voting, stacking. The recovered scripts used **RF + DT + KNN** for title-only hard voting and **RF + ANN + KNN** for combined hard voting; soft/weighted-soft/stacking use RF + ANN + LR, with LR as stacking meta-estimator. The article's summary of combined ensemble base learners is less specific/different; this package follows the recovered code and documents that distinction.

## Dataset interface

Expected pickle entries:

| Key | Contents |
|---|---|
| `train_resampled_title` | `(X_train, y_train_lp)` |
| `test_tfidf_title` | `(X_test, y_test_lp)` |
| `train_resampled_combined` | `(X_train, y_train_lp)` |
| `test_tfidf_combined` | `(X_test, y_test_lp)` |

The workbook must include `Label_Powerset_ID` and `CWE_Label`. The latter stores a serialized list of CWE labels. All input data are loaded from `data/`, independent of the current working directory.

## Paper comparison

REFSQ manuscript Table 3 reports (micro / macro / weighted F1):

| Model | Title | Title + description |
|---|---|---|
| DT | .17 / .15 / .18 | .16 / .13 / .16 |
| RF | .19 / .16 / .19 | .19 / .15 / .18 |
| SVM | .17 / .13 / .17 | .19 / .15 / .18 |
| LR | .18 / .14 / .18 | .18 / .15 / .18 |
| KNN | .18 / .14 / .18 | .21 / .17 / .20 |
| NB | .14 / .11 / .15 | .17 / .12 / .17 |
| ANN | .21 / .17 / .21 | .19 / .15 / .18 |
| Hard voting | .17 / .14 / .18 | .21 / .17 / .20 |
| Soft voting | .20 / .16 / .20 | .20 / .16 / .20 |
| Weighted soft | .19 / .16 / .19 | .20 / .17 / .20 |
| Stacking | .21 / .16 / .21 | .19 / .16 / .19 |

These are **reference values from the manuscript**, not verified outputs of the reconstructed scripts. Check the generated `summary.csv` against them before stating that results were reproduced.

## Publication considerations

The pickle is large (over 200 MB) and exceeds GitHub's normal 100 MB per-file Git limit. Host it through Git LFS or an external research data repository (e.g., Zenodo), and link it from the README if separating data from code. Review licensing and redistribution rights before public release. If GitHub rejects the push, **do not** commit the large pickle to ordinary Git history.

The article's current Data Availability Statement mentions function-level mapping artifacts that are **not included here**; update it to accurately describe this ML-only release.

## Research integrity

The `original_sources/` files are preserved without modification. The portable `scripts/` are a documented reconstruction and contain corrections; they should not be represented as the exact source files used to generate historical manuscript tables. This package has undergone syntax checks and dataset-interface checks, not a full eleven-model-by-two-input reproduction run.
