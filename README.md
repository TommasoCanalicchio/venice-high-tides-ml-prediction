# 🌊 High Water — Predicting Venice Acqua Alta

A machine learning project to predict **high tides (acqua alta) in Venice** using meteorological and oceanographic data collected from weather stations across the Venetian lagoon.

---

## 📌 Project Overview

Venice's iconic acqua alta events occur when tidal levels exceed critical thresholds, causing flooding across the city. This project tackles the challenge as a **binary classification problem**: given meteorological readings at a point in time, predict whether a high tide will be observed **6 hours later**.

- **Target variable**: `y = 2` if a high tide occurs 6 hours after the observation, `y = 1` otherwise
- **Class imbalance**: High tides represent fewer than 8% of observations
- **Dataset**: 1,000 labelled training observations and 866 unlabelled test observations, each with 37 variables (meteorological sensors from multiple lagoon stations)


---

## ⚙️ Methodology

### 1. Data Preprocessing
- Dropped `Piattaforma_CNR_Solar_Radiation` due to ~50% missing values
- Imputed remaining missing values (~20 rows) using **KNN Imputer** with 10 neighbours
- Analysed inter-variable correlations; identified high collinearity among same-type sensors (tide levels, wind speeds, etc.)
- Engineered 5 aggregate features (`Average_Tide_Level`, `Average_Wind_Speed`, `Average_Solar_Radiation`, `First_2_Air_Temperature`, `Average_Pressure`) as an alternative reduced feature set

### 2. Models Explored
Three ensemble classifiers were evaluated on both the full and the reduced feature sets:
- Random Forest
- Gradient Boosting
- **AdaBoost** ✅ *(best performer)*

Ensemble methods were preferred for their natural robustness to class imbalance.

### 3. Hyperparameter Tuning
**Randomized Search Cross-Validation** (100 iterations, 5-fold CV) was used to search the hyperparameter space, guided by a **custom loss function** that penalises false negatives (missed high tides) five times more heavily than false positives — reflecting the real-world cost asymmetry.

### 4. Best Model — AdaBoost

| Hyperparameter | Optimal Value |
|---|---|
| `n_estimators` | 87 |
| `learning_rate` | 1.33646 |
| `base_estimator max_depth` | 2 |

Final score on the held-out evaluation platform: **0.10** (custom penalty metric, lower is better).

---

## 🔍 Key Findings

Feature importance analysis revealed the most influential predictors (after astronomical tide):

| Feature | Importance |
|---|---|
| Astronomical_Tide+6 | 0.123 |
| Misericordia_Average_Wind_Direction | 0.064 |
| Faro_Diga_Lido_Average_Wind_Direction | 0.046 |
| Misericordia_Significant_Wave_Height | 0.045 |
| Piattaforma_CNR_Water_Temperature | 0.045 |
| San_Giorgio_Humidity | 0.041 |

Wind direction, wave height, water temperature, and humidity are recommended as priority monitoring variables for tidal forecasting.

---

## 📄 Context

This project was developed for a machine learning data challenge (2023). Predictions on the unlabelled test set were submitted to an evaluation platform that returned the score on the custom penalty metric.

---
