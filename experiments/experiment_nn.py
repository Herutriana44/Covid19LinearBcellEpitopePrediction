import pandas as pd
import sys
sys.path.append('.')
from models_logic.models import nn

if __name__ == "__main__":
    df = pd.read_csv('dataset/dataset_type_2_vers2_hidropobicity.csv')
    prop_scales = ['Kyte-Doolittle', 'Hopp-Woods', 'Cornette', 'Eisenberg', 'Rose', 'Janin', 'Engelman GES']
    test_sizes = [0.1, 0.2, 0.3]
    results = []
    for prop in prop_scales:
        for size in test_sizes:
            acc, auc = nn(df, ['Position z-score', prop], 'label', size)
            results.append({'prop': prop, 'test_size': size, 'accuracy': acc, 'auc': auc})
    pd.DataFrame(results).to_csv('results/nn_results.csv', index=False)
