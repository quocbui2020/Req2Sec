# Machine Learning Replication Package for Vulnerability Classification of Software Requirements

This repository provides the datasets, Python scripts, and instructions needed to run the machine learning experiments for predicting Common Weakness Enumeration (CWE) classes from software requirement text. It covers **title-only** and **title-plus-description** inputs, using baseline classifiers and ensemble models.

## Repository contents

```text
.
├── README.md
├── requirements.txt
├── scripts/
│   ├── check_dataset.py
│   ├── common.py
│   └── run_experiments.py
├── original_sources/
│   └── [individual baseline and ensemble model scripts]
└── data/
    ├── Original_Requirement_to_Vulnerability_Mapping.xlsx
    └── label_powerset_filtered_for_smote_with_updated_data.xlsx
```

The processed dataset (`processed_data_all.pkl`, approximately 200 MB) is hosted separately on Zenodo to keep the source code repository lightweight. Download the dataset from [10.5281/zenodo.24267927](https://zenodo.org/records/24267928) and place the file in the `data/` directory before running the experiments.

### Dataset files

- **`data/Original_Requirement_to_Vulnerability_Mapping.xlsx`**: The original requirement-to-vulnerability (R-to-V) mapping dataset used as the source for the ML dataset preparation. It contains **66,920 mapping records**, covering **27,522 unique requirements** and **1,442 unique CVEs**. Its columns are `CVE_ID`, `CWE_ID`, `Description`, `CVSS_Metric_BaseScores`, `Requirement_ID`, `Requirement_Title`, `Requirement_Description`, and `Unique_Hash`. This workbook provides the original trace links; it is **not** read directly by the experiment runner.
- **`data/processed_data_all.pkl`**: The processed dataset used for the machine learning experiments. It contains resampled training features and labels, held-out test features and labels, and fitted TF-IDF vectorizers for both input configurations. The feature matrices and targets are ready for model training and evaluation; the upstream data collection and preprocessing stages do not need to be rerun.
- **`data/label_powerset_filtered_for_smote_with_updated_data.xlsx`**: Maps Label Powerset class IDs to their corresponding CWE label sets, used to interpret model predictions.

### Code

- **`scripts/check_dataset.py`** checks that the required dataset files are present and can be loaded.
- **`scripts/run_experiments.py`** runs the selected classifiers and computes evaluation metrics.
- **`scripts/common.py`** contains shared data-loading and evaluation functions.
- **`original_sources/`** contains individual model implementations for reference. Use `scripts/run_experiments.py` for the commands below.

## Requirements

Use Python 3.10–3.12. Install the dependencies listed in `requirements.txt`. The serialized TF-IDF vectorizers were created with scikit-learn 1.6.1, so using that version is recommended for compatibility.

The training matrices are large; some models require substantial RAM and processing time.

## Setup

From the repository's root directory:

```bash
python -m venv .venv
```

Activate the virtual environment:

**macOS/Linux**

```bash
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies and check the dataset:

```bash
python -m pip install -r requirements.txt
python scripts/check_dataset.py
```

Ensure that `processed_data_all.pkl` and `label_powerset_filtered_for_smote_with_updated_data.xlsx` are in `data/`. The R-to-V mapping workbook is provided for inspection of the source traceability data and is not required by the model runner.

## Run the experiments

Run all baseline and ensemble models on both input configurations:

```bash
python scripts/run_experiments.py
```

Run individual experiment groups:

```bash
# Title-only baseline models
python scripts/run_experiments.py --config title --family baselines

# Title-only ensemble models
python scripts/run_experiments.py --config title --family ensembles

# Title + description baseline models
python scripts/run_experiments.py --config combined --family baselines

# Title + description ensemble models
python scripts/run_experiments.py --config combined --family ensembles
```

Run a single model, for example a decision tree on title-only features:

```bash
python scripts/run_experiments.py --config title --family baselines --models dt
```

Supported model names:

| Baselines | Ensembles |
|---|---|
| `dt` (decision tree), `rf` (random forest), `nb` (Gaussian Naive Bayes), `ann` (MLP), `svm` (linear SVM), `lr` (logistic regression), `knn` (k-nearest neighbors) | `hard`, `soft`, `weighted_soft`, `stacking` |

The runner computes evaluation metrics on the held-out test set and writes generated outputs locally. These outputs are not included in the repository. To choose the local output location, use `--output PATH`.

Optional five-fold cross-validation can be requested with `--cv`. This operates on the already resampled training data, so it should not be interpreted as leakage-free model selection.

## Data format

The R-to-V workbook contains one row per requirement–CVE mapping record. It serves as the source mapping dataset, whereas the pickle contains the numerical representations and labels used directly for model training and testing. The model runner uses the pickle and Label Powerset workbook; it does not rebuild the ML dataset from the original R-to-V workbook.

The pickle contains these primary entries:

| Key | Value |
|---|---|
| `train_resampled_title` | Title-only training feature matrix and Label Powerset targets |
| `test_tfidf_title` | Title-only held-out test feature matrix and targets |
| `train_resampled_combined` | Title + description training feature matrix and targets |
| `test_tfidf_combined` | Title + description held-out test feature matrix and targets |

The Excel workbook provides the `Label_Powerset_ID` and `CWE_Label` columns used to decode the targets into CWE label sets.

**Security note:** Python pickle files can execute code when loaded. Load the dataset only from the trusted distribution link associated with this repository.
