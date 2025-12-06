#!/usr/bin/env python
"""
Extended Benchmark: MotherNet vs TabICL vs TabPFN vs CatBoost vs XGBoost

Comprehensive comparison of tabular classification models.
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
    from ticl.prediction import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    print("WARNING: TabPFN not available")
    TABPFN_AVAILABLE = False

try:
    from tabicl import TabICLClassifier
    TABICL_AVAILABLE = True
except ImportError:
    print("WARNING: TabICL not available")
    TABICL_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    print("WARNING: CatBoost not available")
    CATBOOST_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("WARNING: XGBoost not available")
    XGBOOST_AVAILABLE = False


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
    """Run full benchmark comparing all models."""
    print("=" * 80)
    print("EXTENDED BENCHMARK: MotherNet vs TabICL vs TabPFN vs CatBoost vs XGBoost")
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
        print("  [1/5] Testing MotherNet...")
        mothernet = MotherNetClassifier(device=device)
        result = benchmark_model(mothernet, X_train, X_test, y_train, y_test, 'MotherNet')
        result['dataset'] = dataset['name']
        result['dataset_key'] = dataset_key
        all_results.append(result)
        if result['error'] is None:
            print(f"    ✓ Accuracy: {result['accuracy']:.4f}, Time: {result['total_time']:.3f}s")

        # Test TabPFN
        if TABPFN_AVAILABLE:
            print("  [2/5] Testing TabPFN...")
            try:
                tabpfn = TabPFNClassifier(device=device)
                result = benchmark_model(tabpfn, X_train, X_test, y_train, y_test, 'TabPFN')
                result['dataset'] = dataset['name']
                result['dataset_key'] = dataset_key
                all_results.append(result)
                if result['error'] is None:
                    print(f"    ✓ Accuracy: {result['accuracy']:.4f}, Time: {result['total_time']:.3f}s")
            except Exception as e:
                print(f"    ✗ TabPFN failed: {e}")
        else:
            print("  [2/5] Skipping TabPFN (not available)")

        # Test TabICL
        if TABICL_AVAILABLE:
            print("  [3/5] Testing TabICL...")
            tabicl = TabICLClassifier()
            result = benchmark_model(tabicl, X_train, X_test, y_train, y_test, 'TabICL')
            result['dataset'] = dataset['name']
            result['dataset_key'] = dataset_key
            all_results.append(result)
            if result['error'] is None:
                print(f"    ✓ Accuracy: {result['accuracy']:.4f}, Time: {result['total_time']:.3f}s")
        else:
            print("  [3/5] Skipping TabICL (not available)")

        # Test CatBoost
        if CATBOOST_AVAILABLE:
            print("  [4/5] Testing CatBoost...")
            catboost = CatBoostClassifier(iterations=100, verbose=0, random_state=random_state)
            result = benchmark_model(catboost, X_train, X_test, y_train, y_test, 'CatBoost')
            result['dataset'] = dataset['name']
            result['dataset_key'] = dataset_key
            all_results.append(result)
            if result['error'] is None:
                print(f"    ✓ Accuracy: {result['accuracy']:.4f}, Time: {result['total_time']:.3f}s")
        else:
            print("  [4/5] Skipping CatBoost (not available)")

        # Test XGBoost
        if XGBOOST_AVAILABLE:
            print("  [5/5] Testing XGBoost...")
            xgb = XGBClassifier(n_estimators=100, random_state=random_state, verbosity=0)
            result = benchmark_model(xgb, X_train, X_test, y_train, y_test, 'XGBoost')
            result['dataset'] = dataset['name']
            result['dataset_key'] = dataset_key
            all_results.append(result)
            if result['error'] is None:
                print(f"    ✓ Accuracy: {result['accuracy']:.4f}, Time: {result['total_time']:.3f}s")
        else:
            print("  [5/5] Skipping XGBoost (not available)")

        print()

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
        'accuracy': ['mean', 'std', 'min', 'max'],
        'balanced_accuracy': ['mean', 'std'],
        'fit_time': ['mean', 'std'],
        'predict_time': ['mean', 'std'],
        'total_time': ['mean', 'std']
    })
    print(summary.to_string())
    print()

    # Ranking table
    print("\nModel Rankings (by average accuracy):")
    print()
    rankings = df_success.groupby('model')['accuracy'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    rankings['rank'] = range(1, len(rankings) + 1)
    rankings.columns = ['Avg Accuracy', 'Std Dev', 'Rank']
    print(rankings[['Rank', 'Avg Accuracy', 'Std Dev']].to_string())
    print()

    print("\nModel Rankings (by average speed):")
    print()
    speed_rankings = df_success.groupby('model')['total_time'].agg(['mean', 'std']).sort_values('mean', ascending=True)
    speed_rankings['rank'] = range(1, len(speed_rankings) + 1)
    speed_rankings.columns = ['Avg Time (s)', 'Std Dev', 'Rank']
    print(speed_rankings[['Rank', 'Avg Time (s)', 'Std Dev']].to_string())
    print()

    # Detailed results table
    print("\nDetailed Results by Dataset:")
    print()

    # Pivot table for better visualization
    pivot_acc = df_success.pivot_table(values='accuracy', index='dataset', columns='model', aggfunc='mean')
    pivot_time = df_success.pivot_table(values='total_time', index='dataset', columns='model', aggfunc='mean')

    print("Accuracy by Dataset:")
    print(pivot_acc.to_string(float_format=lambda x: f'{x:.4f}'))
    print()

    print("Total Time (s) by Dataset:")
    print(pivot_time.to_string(float_format=lambda x: f'{x:.3f}'))
    print()

    # Win summary
    print("\nWin Summary (Best accuracy per dataset):")
    print()
    for dataset in df_success['dataset'].unique():
        df_dataset = df_success[df_success['dataset'] == dataset]
        best_model = df_dataset.loc[df_dataset['accuracy'].idxmax(), 'model']
        best_acc = df_dataset['accuracy'].max()
        print(f"  {dataset:30s} → {best_model:15s} ({best_acc:.4f})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extended Benchmark: All Models")
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--test-size', type=float, default=0.33,
                        help='Test set size (default: 0.33)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')
    parser.add_argument('--output', type=str, default='benchmark_extended_results.csv',
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
