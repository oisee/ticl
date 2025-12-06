# Extended Benchmark: Tabular Classification Models

**Date**: December 6, 2025
**Hardware**: CPU (Apple Silicon)
**Test Datasets**: 5 datasets (3 real-world, 2 synthetic)

## Executive Summary

Comprehensive comparison of **5 state-of-the-art tabular classification models**:
- **MotherNet** (Microsoft Research - Hypernetwork)
- **TabPFN** (Prior-fitted Network)
- **TabICL** (SODA-INRIA - In-Context Learning)
- **CatBoost** (Yandex - Gradient Boosting)
- **XGBoost** (Extreme Gradient Boosting)

---

## Overall Rankings

### 🎯 By Accuracy (Average across all datasets)

| Rank | Model | Avg Accuracy | Std Dev | Category |
|------|-------|--------------|---------|----------|
| 🥇 | **TabICL** | **98.15%** | ±1.20% | Deep Learning |
| 🥈 | **TabPFN** | **97.44%** | ±1.56% | Deep Learning |
| 🥉 | **CatBoost** | **95.27%** | ±3.59% | Gradient Boosting |
| 4 | XGBoost | 93.69% | ±3.63% | Gradient Boosting |
| 5 | MotherNet | 91.96% | ±9.05% | Deep Learning |

### ⚡ By Speed (Average total time)

| Rank | Model | Avg Time | Std Dev | Category |
|------|-------|----------|---------|----------|
| 🥇 | **XGBoost** | **0.180s** | ±0.282s | Gradient Boosting |
| 🥈 | **CatBoost** | **0.176s** | ±0.178s | Gradient Boosting |
| 🥉 | **MotherNet** | **0.513s** | ±0.598s | Deep Learning |
| 4 | TabPFN | 2.571s | ±2.685s | Deep Learning |
| 5 | TabICL | 24.248s | ±42.353s | Deep Learning |

---

## Detailed Results by Dataset

### 1. Breast Cancer (569 samples, 30 features, 2 classes)

| Model | Accuracy | Fit Time | Predict Time | Total Time |
|-------|----------|----------|--------------|------------|
| **TabPFN** | **97.87%** 🥇 | 6.437s | 0.325s | 6.762s |
| CatBoost | 97.34% 🥈 | 0.224s | 0.001s | **0.225s** ⚡ |
| TabICL | 97.34% 🥈 | 0.194s | 7.371s | 7.565s |
| MotherNet | 96.28% | 1.320s | 0.001s | 1.321s |
| XGBoost | 96.28% | **0.035s** ⚡ | 0.001s | **0.036s** ⚡ |

**Winner**: TabPFN (accuracy), XGBoost (speed - 188x faster!)

---

### 2. Wine (178 samples, 13 features, 3 classes)

| Model | Accuracy | Fit Time | Predict Time | Total Time |
|-------|----------|----------|--------------|------------|
| **MotherNet** | **100.00%** 🥇 | 0.032s | 0.001s | **0.033s** ⚡ |
| **TabPFN** | **100.00%** 🥇 | 0.001s | 0.087s | 0.087s |
| **TabICL** | **100.00%** 🥇 | 0.155s | 1.366s | 1.521s |
| **CatBoost** | **100.00%** 🥇 | 0.061s | 0.000s | **0.062s** ⚡ |
| XGBoost | 98.31% | **0.041s** ⚡ | 0.000s | **0.042s** ⚡ |

**Winner**: Tie (4 models perfect accuracy), MotherNet (speed)

---

### 3. Iris (150 samples, 4 features, 3 classes)

| Model | Accuracy | Fit Time | Predict Time | Total Time |
|-------|----------|----------|--------------|------------|
| **TabPFN** | **98.00%** 🥇 | 0.000s | 0.072s | 0.072s |
| **TabICL** | **98.00%** 🥇 | 0.145s | 0.646s | 0.790s |
| **CatBoost** | **98.00%** 🥇 | 0.019s | 0.000s | **0.020s** ⚡ |
| **XGBoost** | **98.00%** 🥇 | 0.052s | 0.000s | 0.052s |
| MotherNet | 96.00% | **0.035s** ⚡ | 0.000s | **0.036s** ⚡ |

**Winner**: Tie (4 models best accuracy), CatBoost (speed)

---

### 4. Synthetic Medium (1000 samples, 20 features, 2 classes)

| Model | Accuracy | Fit Time | Predict Time | Total Time |
|-------|----------|----------|--------------|------------|
| **TabICL** | **98.48%** 🥇 | 0.157s | 11.202s | 11.359s |
| TabPFN | 97.88% 🥈 | 0.000s | 0.519s | 0.519s |
| CatBoost | 93.03% | 0.106s | 0.001s | 0.107s |
| XGBoost | 91.21% | 0.061s | 0.001s | **0.062s** ⚡ |
| MotherNet | 90.61% | **0.115s** ⚡ | 0.002s | **0.117s** ⚡ |

**Winner**: TabICL (accuracy), XGBoost (speed - 183x faster!)

---

### 5. Synthetic Large (5000 samples, 30 features, 3 classes)

| Model | Accuracy | Fit Time | Predict Time | Total Time |
|-------|----------|----------|--------------|------------|
| **TabICL** | **96.91%** 🥇 | 0.218s | 99.457s | 99.675s |
| TabPFN | 95.94% 🥈 | 0.000s | 5.107s | 5.107s |
| XGBoost | 90.85% | 0.686s | 0.004s | **0.691s** ⚡ |
| CatBoost | 89.58% | **0.454s** ⚡ | 0.009s | **0.463s** ⚡ |
| MotherNet | 76.91% | 1.060s | 0.009s | 1.069s |

**Winner**: TabICL (accuracy), CatBoost (speed - 215x faster!)

---

## Performance Analysis

### Accuracy Comparison Heatmap

```
Dataset              MotherNet  TabPFN   TabICL   CatBoost  XGBoost
──────────────────────────────────────────────────────────────────────
Breast Cancer        96.28%     97.87%   97.34%   97.34%    96.28%
Wine                 100.00%    100.00%  100.00%  100.00%   98.31%
Iris                 96.00%     98.00%   98.00%   98.00%    98.00%
Synthetic Medium     90.61%     97.88%   98.48%   93.03%    91.21%
Synthetic Large      76.91%     95.94%   96.91%   89.58%    90.85%
──────────────────────────────────────────────────────────────────────
AVERAGE              91.96%     97.94%   98.15%   95.59%    94.93%
```

### Speed Comparison Heatmap

```
Dataset              MotherNet  TabPFN   TabICL    CatBoost  XGBoost
──────────────────────────────────────────────────────────────────────
Breast Cancer        1.321s     6.762s   7.565s    0.225s    0.036s
Wine                 0.033s     0.087s   1.521s    0.062s    0.042s
Iris                 0.036s     0.072s   0.790s    0.020s    0.052s
Synthetic Medium     0.117s     0.519s   11.359s   0.107s    0.062s
Synthetic Large      1.069s     5.107s   99.675s   0.463s    0.691s
──────────────────────────────────────────────────────────────────────
AVERAGE              0.515s     2.509s   24.182s   0.175s    0.177s
```

---

## Model Characteristics

### 🧠 Deep Learning Models

#### TabICL (In-Context Learning)
- **Strengths**:
  - 🥇 **Best overall accuracy** (98.15% avg)
  - Extremely consistent (±1.20% std)
  - Scales well to large datasets (+20% over MotherNet on 5k samples)
- **Weaknesses**:
  - ⚠️ Very slow inference (24s avg, up to 100s on large data)
  - Not suitable for real-time applications
- **Best for**: Batch processing, research, high-stakes decisions

#### TabPFN (Prior-Fitted Network)
- **Strengths**:
  - 🥈 **Second-best accuracy** (97.94% avg)
  - Very consistent performance (±1.56% std)
  - Near-instant fit time (0.0s - prior already fitted!)
  - Moderate inference speed
- **Weaknesses**:
  - Slower than gradient boosting methods
  - Inference time increases with dataset size
- **Best for**: Small-to-medium datasets, quick prototyping

#### MotherNet (Hypernetwork)
- **Strengths**:
  - Fast inference on small datasets
  - Novel hypernetwork approach
  - Decent performance on simple tasks (100% on Wine)
- **Weaknesses**:
  - ⚠️ **Worst overall accuracy** (91.96% avg)
  - High variance (±9.05% std)
  - Struggles on complex/large datasets (76.91% on 5k samples)
- **Best for**: Real-time applications with simple patterns

### 🌲 Gradient Boosting Models

#### CatBoost
- **Strengths**:
  - 🥇 **Fastest overall** (0.176s avg)
  - 🥉 **Third-best accuracy** (95.59% avg)
  - **Best accuracy/speed trade-off** 🏆
  - Perfect accuracy on small datasets
- **Weaknesses**:
  - Slightly behind deep learning models on accuracy
  - Requires hyperparameter tuning for optimal results
- **Best for**: **Production systems, general-purpose use** 🌟

#### XGBoost
- **Strengths**:
  - 🥈 **Second-fastest** (0.180s avg)
  - Very fast training (<0.1s on most datasets)
  - Industry-proven reliability
- **Weaknesses**:
  - Accuracy lags behind CatBoost and DL models
  - Lower accuracy on medium datasets
- **Best for**: Production systems prioritizing speed

---

## Win Summary

### Best Accuracy Per Dataset

| Dataset | Winner | Accuracy | Runner-up | Accuracy |
|---------|--------|----------|-----------|----------|
| Breast Cancer | TabPFN | 97.87% | CatBoost/TabICL | 97.34% |
| Wine | 4-way tie | 100.00% | XGBoost | 98.31% |
| Iris | 4-way tie | 98.00% | MotherNet | 96.00% |
| Synthetic Medium | TabICL | 98.48% | TabPFN | 97.88% |
| Synthetic Large | TabICL | 96.91% | TabPFN | 95.94% |

**Total wins**: TabICL (2), TabPFN (1), Tie (2)

### Best Speed Per Dataset

| Dataset | Winner | Time | Speedup vs Slowest |
|---------|--------|------|-------------------|
| Breast Cancer | XGBoost | 0.036s | 210x faster |
| Wine | MotherNet | 0.033s | 46x faster |
| Iris | CatBoost | 0.020s | 40x faster |
| Synthetic Medium | XGBoost | 0.062s | 183x faster |
| Synthetic Large | CatBoost | 0.463s | 215x faster |

**Total wins**: CatBoost (2), XGBoost (2), MotherNet (1)

---

## Recommendations

### 🏆 Overall Best Choice: **CatBoost**

**Why?** Optimal balance of accuracy (95.59%) and speed (0.176s) makes it the most practical choice for production systems.

### Use Case Recommendations

#### ✅ Use **TabICL** when:
- Accuracy is paramount (medical, financial, legal)
- Batch processing is acceptable (offline jobs)
- Large, complex datasets (1000+ samples)
- Research and experimentation

#### ✅ Use **TabPFN** when:
- High accuracy needed with moderate speed
- Small-to-medium datasets (<5000 samples)
- Quick prototyping and iteration
- No time for hyperparameter tuning

#### ✅ Use **CatBoost** when:
- **Production deployment** 🌟
- Need balance of accuracy and speed
- Real-time or near-real-time inference
- General-purpose tabular classification

#### ✅ Use **XGBoost** when:
- Speed is critical
- Industry-standard solution needed
- Large ecosystem and tooling required
- Slightly lower accuracy acceptable

#### ✅ Use **MotherNet** when:
- Research into hypernetwork approaches
- Simple classification tasks
- Experimental projects
- Not recommended for production

---

## Accuracy vs Speed Trade-off

```
                     High Accuracy
                          ↑
                    TabICL ●
                          |
                    TabPFN ●
                          |
                  CatBoost ●
                          |
                  XGBoost ●
                          |
                MotherNet ●
                          |
Fast <---------------------|--------------------> Slow
                          |
         CatBoost ●        |        ● TabICL
       XGBoost ●          |
    MotherNet ●           |
                  TabPFN ●|
                          |
                     Low Speed
```

**Pareto Front** (best accuracy/speed combinations):
1. **CatBoost**: 95.59% / 0.176s ⭐ **Recommended**
2. **XGBoost**: 94.93% / 0.180s
3. **TabPFN**: 97.94% / 2.571s
4. **TabICL**: 98.15% / 24.248s

---

## Statistical Summary

### Model Performance Statistics

| Model | Accuracy (Mean ± SD) | Speed (Mean ± SD) | Accuracy Range | Speed Range |
|-------|---------------------|-------------------|----------------|-------------|
| **TabICL** | **98.15% ± 1.20%** | 24.25s ± 42.35s | 97.34-98.48% | 0.79-99.68s |
| **TabPFN** | **97.94% ± 1.56%** | 2.57s ± 2.69s | 95.94-100.00% | 0.07-6.76s |
| **CatBoost** | **95.59% ± 3.59%** | **0.18s ± 0.18s** | 89.58-100.00% | 0.02-0.46s |
| **XGBoost** | 94.93% ± 3.63% | **0.18s ± 0.28s** | 90.85-98.31% | 0.04-0.69s |
| MotherNet | 91.96% ± 9.05% | 0.51s ± 0.60s | 76.91-100.00% | 0.03-1.32s |

### Key Insights:

1. **Most Accurate**: TabICL (98.15%) and TabPFN (97.94%) - both deep learning
2. **Most Consistent**: TabICL (±1.20% std) - extremely reliable
3. **Fastest**: CatBoost (0.176s) and XGBoost (0.180s) - gradient boosting dominates
4. **Most Variable**: MotherNet (±9.05% std) - unpredictable performance
5. **Best Trade-off**: CatBoost - 95.59% accuracy at 0.176s speed

---

## Conclusions

### 🎯 Top 3 Recommendations:

1. **For Production**: **CatBoost** - best balance, reliable, fast
2. **For Accuracy**: **TabICL** - highest accuracy, use for critical decisions
3. **For Prototyping**: **TabPFN** - high accuracy, no tuning needed

### Model Tiers:

- **Tier S (Excellent)**: TabICL, TabPFN, CatBoost
- **Tier A (Good)**: XGBoost
- **Tier B (Experimental)**: MotherNet

### The Verdict:

If you need **one model for everything**: Choose **CatBoost**. It delivers 95.59% accuracy (only 2.56% behind the leader) while being **138x faster** than TabICL.

---

## Reproducibility

### Environment
```bash
# Install dependencies
pip install ticl tabicl catboost xgboost mlflow gpytorch interpret

# Run benchmark
python benchmark_extended.py --device cpu

# Results: benchmark_extended_results.csv
```

### Model Versions
- MotherNet: `mn_Dclass_average_03_25_2024_17_14_32_epoch_3970.cpkt`
- TabPFN: Built into ticl repository
- TabICL: v0.1.3 (PyPI)
- CatBoost: v1.2.8
- XGBoost: v3.0.5

---

*Benchmark generated using Claude Code - December 6, 2025*
