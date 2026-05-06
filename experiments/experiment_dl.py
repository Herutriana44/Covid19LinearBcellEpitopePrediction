"""
Eksperimen Deep Learning
"""
import pandas as pd
import os
from algorithms.deep_learning import DLExperiment

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    dataset_path = 'dataset/dataset_type_2_vers2_hidropobicity.csv'
    df = pd.read_csv(dataset_path)
    feature_base = ['Position z-score']
    target = 'label'
    prop_scales = ['Kyte-Doolittle', 'Hopp-Woods', 'Cornette', 'Eisenberg', 'Rose', 'Janin', 'Engelman GES']
    test_sizes = [0.1, 0.2, 0.3]

    results = []
    experiment = DLExperiment(df, feature_base, target)
    for prop_scale in prop_scales:
        for test_size in test_sizes:
            try:
                accuracy, auc = experiment.run(prop_scale, test_size)
                results.append({
                    'algorithm': 'deep_learning',
                    'propensity_scale': prop_scale,
                    'test_size': test_size,
                    'accuracy': accuracy,
                    'auc': auc,
                })
            except Exception as e:
                results.append({
                    'algorithm': 'deep_learning',
                    'propensity_scale': prop_scale,
                    'test_size': test_size,
                    'accuracy': None,
                    'auc': None,
                    'error': str(e),
                })

    pd.DataFrame(results).to_csv('results/dl_experiment_results.csv', index=False)
    print('Deep Learning experiment selesai. Hasil disimpan di results/dl_experiment_results.csv')
