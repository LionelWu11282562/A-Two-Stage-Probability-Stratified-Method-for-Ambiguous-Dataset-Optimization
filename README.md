# PU-Boost

**PU-Boost: A Two-Stage Reliable-Sample Reconstruction Positive-Unlabeled Learning Framework for Screening Undiagnosed Hypertension**

PU-Boost is a two-stage positive-unlabeled (PU) learning framework designed for screening undiagnosed hypertension under incomplete diagnostic labeling.

Previously diagnosed hypertension cases are treated as reliable positives, while individuals without a prior diagnosis form an unlabeled population containing both hidden hypertensive and non-hypertensive participants.

The framework progressively reconstructs a high-confidence training set through:

- Stage 1: Elastic Net-based global P/U ranking
- Stage 2: Random Forest refinement of uncertain samples
- Rejection of highly ambiguous observations
- Final XGBoost training on the reconstructed balanced dataset

<p align="center">
  <img src="figures/pu_boost_framework.png" width="900">
</p>

<p align="center">
  <b>Overview of the two-stage PU-Boost reliable-sample reconstruction framework.</b>
</p>

## Key Results

In the independent test set, PU-Boost achieved:

- Sensitivity: **0.817**
- Balanced accuracy: **0.682**
- False negatives: **729**
- 41.1% fewer false negatives than standard XGBoost

The method prioritizes high-sensitivity screening and reduces missed hypertension cases at the cost of lower specificity.

## Method Overview

PU-Boost consists of three main components:

1. **Global stratification** using Elastic Net
2. **Nonlinear refinement** using Random Forest
3. **Final classification** using XGBoost

Highly uncertain samples that remain ambiguous after both stages are excluded from final supervised training.

## Interpretation

SHAP analysis identified age, BMI, waist circumference, family history of hypertension, and related clinical factors as major contributors to prediction.

## Authors

- Huanqi Wu — University of Arizona
- Liangkun Shi — University of Arizona
- Hanfei Yang — University of Arizona
- Lizhu Guo — Beijing Anzhen Hospital, Capital Medical University

Correspondence: huanqiwu@arizona.edu
