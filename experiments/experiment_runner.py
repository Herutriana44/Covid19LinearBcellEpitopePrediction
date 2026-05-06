"""
Main experiment runner script
Runs all classification algorithms and exports results to CSV
"""
import pandas as pd
import os
from pathlib import Path
import sys

# Import all algorithm modules
from algorithms.deep_learning import DLExperiment
from algorithms.neural_network import NNExperiment
from algorithms.random_forest import RFExperiment
from algorithms.decision_tree import DTExperiment
from algorithms.svm import SVMExperiment
from algorithms.hmm import HMMExperiment


def run_all_experiments(dataset_path='dataset/dataset_type_2_vers2_hidropobicity.csv'):
    """
    Run all experiments and collect results
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Configuration
    algorithms = ['deep_learning', 'neural_network', 'random_forest', 'decision_tree', 'svm', 'hmm']
    prop_scales = ['Kyte-Doolittle', 'Hopp-Woods', 'Cornette', 'Eisenberg', 'Rose', 'Janin', 'Engelman GES']
    test_sizes = [0.1, 0.2, 0.3]
    
    # Features and target
    feature_base = ['Position z-score']
    target = 'label'
    
    # Initialize results dataframe
    results = []
    
    print(f"Starting experiments...")
    print(f"Total experiments: {len(algorithms) * len(prop_scales) * len(test_sizes)}")
    
    # Run each algorithm experiment
    for algo_name in algorithms:
        print(f"\n{'='*60}")
        print(f"Running {algo_name.upper()} experiments...")
        print(f"{'='*60}")
        
        if algo_name == 'deep_learning':
            experiment = DLExperiment(df, feature_base, target)
        elif algo_name == 'neural_network':
            experiment = NNExperiment(df, feature_base, target)
        elif algo_name == 'random_forest':
            experiment = RFExperiment(df, feature_base, target)
        elif algo_name == 'decision_tree':
            experiment = DTExperiment(df, feature_base, target)
        elif algo_name == 'svm':
            experiment = SVMExperiment(df, feature_base, target)
        elif algo_name == 'hmm':
            experiment = HMMExperiment(df, feature_base, target)
        
        # Run experiments for each propensity scale and test size
        for prop_scale in prop_scales:
            for test_size in test_sizes:
                try:
                    print(f"  Running: {algo_name} | {prop_scale} | test_size={test_size}")
                    accuracy, auc = experiment.run(prop_scale, test_size)
                    
                    results.append({
                        'algorithm': algo_name,
                        'propensity_scale': prop_scale,
                        'test_size': test_size,
                        'accuracy': accuracy,
                        'auc': auc,
                        'n_amino_acids': 1
                    })
                    
                    print(f"    ✓ Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
                    
                except Exception as e:
                    print(f"    ✗ Error: {str(e)}")
                    results.append({
                        'algorithm': algo_name,
                        'propensity_scale': prop_scale,
                        'test_size': test_size,
                        'accuracy': None,
                        'auc': None,
                        'n_amino_acids': 1
                    })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_path = 'results/classification_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(results_df.groupby('algorithm')[['accuracy', 'auc']].agg(['mean', 'std', 'min', 'max']))
    
    return results_df


if __name__ == '__main__':
    results = run_all_experiments()
    print(f"\nExperiment run completed successfully!")
