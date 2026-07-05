# Assignment 03 — Predicting Used-Car Fair Market Price

**Course:** EECE 6544 — Introduction to Machine Learning and Pattern Recognition (Summer 2026)

Noah Caney and Aakarsh Arora

---

## Overview

This project builds a regression model that predicts the fair market price of a used car
from its attributes (make, year, kilometers driven, fuel type, transmission, owner history,
engine specs, location, and body dimensions).

It is based on the WheelsBazaar scenario: sellers currently set their own prices, manual review
doesn't scale, and an upcoming instant-buy program needs a trustworthy automated price estimate.
An accurate model helps flag overpriced and underpriced listings and gives the instant-buy
program a starting price.

The notebook follows a standard supervised-learning workflow:

1. Load and explore the data
2. Clean the text "spec" columns (Engine, Max Power, Max Torque) into numbers
3. Select useful columns and handle missing values
4. One-hot encode the categorical features
5. Split into training (70%) and test (30%) sets
6. Fit a Linear Regression model and evaluate with R², MAE, and RMSE
7. Apply Lasso (L1) and Ridge (L2) regularization to reduce overfitting

## Dataset

The dataset (`car_details_v4.csv`) contains about 2,000 used-car listings with 20 columns,
including the target column `Price`.

It is already included in this repository, so no separate download is required. If you ever need
to obtain it again, it is a public used-car listings dataset ("Car details v4") available on
Kaggle — search for "Car details v4" and download the CSV, then place it in the same folder as
the notebook (or upload it to Colab as described below).

## How to run

### Google Colab

1. Open `Assignment_03_Car_Price_Prediction.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Click the folder icon in the left sidebar, then upload `car_details_v4.csv`.
3. Run the cells from top to bottom (Runtime → Run all).

## Summary of findings

| Model | Train R² | Test R² |
|-------|:--------:|:-------:|
| Linear Regression | 0.82 | 0.73 |
| Lasso (L1) | 0.80 | 0.71 |
| Ridge (L2) | 0.72 | 0.65 |

- The plain Linear Regression model explains about 73% of the variation in used-car prices
  on unseen data, with a mean absolute error (MAE) of roughly ₹680,000 and an RMSE of about
  ₹1.23M. Features such as year, kilometers, engine size, power, and brand carry clear signal.
- The training R² (0.82) is higher than the test R² (0.73), which is the expected sign of mild
  overfitting.
- Lasso and Ridge trade a small amount of training accuracy for better generalization, which is
  the behavior described in the regularization portion of the lecture. At `alpha = 50` the
  regularized models are close to the baseline here, since the dataset does not overfit severely.
- Possible next steps: tune `alpha` with cross-validation, predict `log(Price)` since prices are
  highly skewed, or try a tree-based model (Random Forest / XGBoost) for higher accuracy.
