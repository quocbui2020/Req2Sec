"""Check that both archived feature configurations and label mappings can be loaded."""
from common import load_data
for config in ('title', 'combined'):
    x_train, y_train, x_test, y_test, mapping, mlb, y_true = load_data(config)
    print(f'{config}: train={x_train.shape}, test={x_test.shape}, CWE labels={len(mlb.classes_)}, LP IDs={len(mapping)}')
print('Dataset compatibility checks passed.')
