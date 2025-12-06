#!/usr/bin/env python
"""
Benchmark comparison: MotherNet vs TabICL

Compares performance and speed of MotherNet and TabICL on various tabular datasets.
"""

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_wine,
    load_iris,
    make_classification,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

warnings.filterwarnings('ignore')

# Import models
from ticl.prediction import MotherNetClassifier
try:
    from tabicl import TabICLClassifier
    TABICL_AVAILABLE = True
except ImportError:
    print("WARNING: TabICL not available")
    TABICL_AVAILABLE = False


def load_datasets():
    """Load various sklearn datasets for benchmarking."""
    datasets = {}

    # Breast Cancer
    X, y = load_breast_cancer(return_X_y=True)
    datasets['breast_cancer'] = {
        'X': X, 'y': y,
        'name': 'Breast Cancer',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y))
    }

    # Wine
    X, y = load_wine(return_X_y=True)
    datasets['wine'] = {
        'X': X, 'y': y,
        'name': 'Wine',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y))
    }

    # Iris
    X, y = load_iris(return_X_y=True)
    datasets['iris'] = {
        'X': X, 'y': y,
        'name': 'Iris',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y))
    }

    # Synthetic medium dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    datasets['synthetic_medium'] = {
        'X': X, 'y': y,
        'name': 'Synthetic Medium (1k samples)',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y))
    }

    # Synthetic large dataset
    X, y = make_classification(
        n_samples=5000,
        n_features=30,
        n_informative=20,
        n_redundant=10,
        n_classes=3,
        random_state=42
    )
    datasets['synthetic_large'] = {
        'X': X, 'y': y,
        'name': 'Synthetic Large (5k samples)',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y))
    }

    return datasets


def benchmark_model(model, X_train, X_test, y_train, y_test, model_name):
    """Benchmark a single model on a dataset."""
    result = {
        'model': model_name,
        'fit_time': None,
        'predict_time': None,
        'total_time': None,
        'accuracy': None,
        'balanced_accuracy': None,
        'f1_score': None,
        'error': None
    }

    try:
        # Fit
        start_time = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start_time
        result['fit_time'] = fit_time

        # Predict
        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_time
        result['predict_time'] = predict_time
        result['total_time'] = fit_time + predict_time

        # Metrics
        result['accuracy'] = accuracy_score(y_test, y_pred)
        result['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)

        # F1 score (handle multiclass)
        n_classes = len(np.unique(y_test))
        if n_classes == 2:
            result['f1_score'] = f1_score(y_test, y_pred)
        else:
            result['f1_score'] = f1_score(y_test, y_pred, average='weighted')

    except Exception as e:
        result['error'] = str(e)
        print(f"  ERROR with {model_name}: {e}")

    return result


def run_benchmark(device='cpu', test_size=0.33, random_state=42):
    """Run full benchmark comparing MotherNet and TabICL."""
    print("=" * 80)
    print("BENCHMARK: MotherNet vs TabICL")
    print("=" * 80)
    print()

    datasets = load_datasets()
    all_results = []

    for dataset_key, dataset in datasets.items():
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset['name']}")
        print(f"  Samples: {dataset['n_samples']}, Features: {dataset['n_features']}, Classes: {dataset['n_classes']}")
        print(f"{'='*80}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            dataset['X'],
            dataset['y'],
            test_size=test_size,
            random_state=random_state
        )

        print(f"  Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        print()

        # Test MotherNet
        print("  Testing MotherNet...")
        mothernet = MotherNetClassifier(device=device)
        result_mothernet = benchmark_model(
            mothernet, X_train, X_test, y_train, y_test, 'MotherNet'
        )
        result_mothernet['dataset'] = dataset['name']
        result_mothernet['dataset_key'] = dataset_key
        all_results.append(result_mothernet)

        if result_mothernet['error'] is None:
            print(f"    Accuracy: {result_mothernet['accuracy']:.4f}")
            print(f"    Fit time: {result_mothernet['fit_time']:.3f}s")
            print(f"    Predict time: {result_mothernet['predict_time']:.3f}s")
            print(f"    Total time: {result_mothernet['total_time']:.3f}s")

        # Test TabICL
        if TABICL_AVAILABLE:
            print("  Testing TabICL...")
            tabicl = TabICLClassifier()
            result_tabicl = benchmark_model(
                tabicl, X_train, X_test, y_train, y_test, 'TabICL'
            )
            result_tabicl['dataset'] = dataset['name']
            result_tabicl['dataset_key'] = dataset_key
            all_results.append(result_tabicl)

            if result_tabicl['error'] is None:
                print(f"    Accuracy: {result_tabicl['accuracy']:.4f}")
                print(f"    Fit time: {result_tabicl['fit_time']:.3f}s")
                print(f"    Predict time: {result_tabicl['predict_time']:.3f}s")
                print(f"    Total time: {result_tabicl['total_time']:.3f}s")

            # Comparison
            if result_mothernet['error'] is None and result_tabicl['error'] is None:
                print()
                print("  Comparison:")
                acc_diff = result_mothernet['accuracy'] - result_tabicl['accuracy']
                time_diff = result_mothernet['total_time'] - result_tabicl['total_time']

                winner_acc = "MotherNet" if acc_diff > 0 else "TabICL" if acc_diff < 0 else "Tie"
                winner_time = "MotherNet" if time_diff < 0 else "TabICL" if time_diff > 0 else "Tie"

                print(f"    Accuracy: {winner_acc} wins (diff: {acc_diff:+.4f})")
                print(f"    Speed: {winner_time} wins (diff: {time_diff:+.3f}s)")
        else:
            print("  Skipping TabICL (not available)")

    # Create summary DataFrame
    df = pd.DataFrame(all_results)

    return df


def print_summary(df):
    """Print summary statistics."""
    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    # Filter successful runs
    df_success = df[df['error'].isna()].copy()

    if len(df_success) == 0:
        print("No successful runs to summarize.")
        return

    # Summary by model
    print("Average Performance by Model:")
    print()
    summary = df_success.groupby('model').agg({
        'accuracy': ['mean', 'std'],
        'balanced_accuracy': ['mean', 'std'],
        'fit_time': ['mean', 'std'],
        'predict_time': ['mean', 'std'],
        'total_time': ['mean', 'std']
    })
    print(summary.to_string())
    print()

    # Detailed results table
    print("\nDetailed Results:")
    print()
    cols_to_show = ['dataset', 'model', 'accuracy', 'balanced_accuracy',
                    'fit_time', 'predict_time', 'total_time']
    print(df_success[cols_to_show].to_string(index=False))

    # Win counts
    if TABICL_AVAILABLE and len(df_success[df_success['model'] == 'TabICL']) > 0:
        print("\n")
        print("Win Summary:")
        print()

        datasets = df_success['dataset'].unique()
        mothernet_wins_acc = 0
        tabicl_wins_acc = 0
        mothernet_wins_speed = 0
        tabicl_wins_speed = 0

        for dataset in datasets:
            df_dataset = df_success[df_success['dataset'] == dataset]
            if len(df_dataset) == 2:
                mothernet_row = df_dataset[df_dataset['model'] == 'MotherNet'].iloc[0]
                tabicl_row = df_dataset[df_dataset['model'] == 'TabICL'].iloc[0]

                if mothernet_row['accuracy'] > tabicl_row['accuracy']:
                    mothernet_wins_acc += 1
                elif tabicl_row['accuracy'] > mothernet_row['accuracy']:
                    tabicl_wins_acc += 1

                if mothernet_row['total_time'] < tabicl_row['total_time']:
                    mothernet_wins_speed += 1
                elif tabicl_row['total_time'] < mothernet_row['total_time']:
                    tabicl_wins_speed += 1

        print(f"  Accuracy wins: MotherNet {mothernet_wins_acc} - {tabicl_wins_acc} TabICL")
        print(f"  Speed wins: MotherNet {mothernet_wins_speed} - {tabicl_wins_speed} TabICL")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark MotherNet vs TabICL")
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--test-size', type=float, default=0.33,
                        help='Test set size (default: 0.33)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')
    parser.add_argument('--output', type=str, default='benchmark_results.csv',
                        help='Output CSV file for results')

    args = parser.parse_args()

    # Run benchmark
    df = run_benchmark(
        device=args.device,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # Print summary
    print_summary(df)

    # Save results
    df.to_csv(args.output, index=False)
    print(f"\n\nResults saved to: {args.output}")
