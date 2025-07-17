# An Interpretable AI Framework for Clear-Sky Detection in Photovoltaics Monitoring

This repository contains the code and resources accompanying the paper:

**"An Interpretable AI Framework for Clear-Sky Detection in Photovoltaics Monitoring"**  
by Bohan Li, Hugo Quest, et al.

![Framework Overview](figures/CSD_Framework.png)

## Overview

This project presents a high-performance, interpretable clear-sky detection (CSD) model designed for photovoltaic (PV) monitoring applications. The framework combines a CatBoost classifier with SHAP (SHapley Additive exPlanations) to analyse and iteratively refine model misclassifications in a closed-loop, explainability-driven workflow.

Key features:
- CatBoost-based CSD classifier trained on 1-minute POA irradiance data
- SHAP-driven diagnostic analysis of false positives and false negatives
- Feature engineering informed by interpretable failure patterns
- Final model achieves **97.3% accuracy**, with a **1.99% false positive rate** and **7.0% false negative rate**

## Repository Contents

- `src/` – Core Python modules for data preprocessing, feature extraction, and model evaluation
- `data/` – Labelled dataset (Jordan & Hansen, 2023)
- `figures/` – Selected visualisations used in the paper
- `requirements.txt` – Python dependencies
- `LICENSE` – MIT License
