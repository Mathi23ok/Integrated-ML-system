# Income Intelligence System (Multi-Model ML Pipeline)

## Overview

Designed and implemented a multi-model machine learning system that predicts:

* Probability of high income (>50K)
* Estimated income (continuous value)
* Income class (Low / Medium / High)

The system integrates regression, classification, and probabilistic modeling into a unified pipeline with deployment capability.

---

## Problem

Given demographic and employment data, generate actionable insights about an individual's income:

* Will they earn high income?
* What is their estimated income?
* Which economic class do they belong to?

---

## System Architecture

```text
Input → Preprocessing → Multi-Model Engine → Unified Output
```

### Models Used

* Logistic Regression → High income probability
* Linear Regression → Estimated income
* Decision Tree Classifier → Income class

---

## Pipeline Design

1. Data Loading and Cleaning
2. Missing Value Handling
3. Feature Encoding (One-Hot Encoding)
4. Feature Engineering

   * Estimated_Income (engineered target)
   * Income_Class (derived classification target)
5. Train-Test Split (randomized)
6. Feature Scaling
7. Model Training (3 models)
8. Evaluation
9. Model Serialization (`system.pkl`)

---

## Results

* Logistic Regression Accuracy: ~83%
* Decision Tree Accuracy: ~80%
* Linear Regression RMSE: realistic (after leakage removal)

---

## Key Technical Challenges

### 1. Data Leakage

Initial model showed near-zero RMSE due to target leakage.
Resolved by removing features used in target construction from training data.

### 2. Overfitting

Decision tree initially achieved ~98% accuracy.
Controlled using `max_depth` to improve generalization.

### 3. Pipeline Consistency

Ensured identical preprocessing for training and inference:

* Same encoding
* Same scaling
* Same feature order

---

## Model Evaluation

### Classification

* Accuracy
* Confusion Matrix
* ROC Curve

### Regression

* RMSE
* Residual Analysis

---

## Key Insights

* Model performance is heavily dependent on feature design
* Data leakage can completely invalidate results
* Multiple models can be combined to produce richer decisions
* Evaluation must go beyond a single metric

---

## Deployment

* Models serialized using `pickle`
* Built FastAPI service for predictions
* Designed Streamlit UI for interaction

---

## Example Output

```json
{
  "estimated_income": 52000,
  "high_income_probability": 0.73,
  "income_class": 1
}
```

---

## Project Structure

```text
system/
├── preprocessing.py
├── train.py
├── system.pkl
├── api.py
├── app.py
```

---

## How to Run

### Train

```bash
python train.py
```

### API

```bash
uvicorn api:app --reload
```

### UI

```bash
streamlit run app.py
```

---

## Limitations

* Regression target is engineered, not real-world income
* Dataset may not reflect real economic complexity
* Limited hyperparameter tuning

---

## Future Improvements

* Use real continuous income dataset
* Add ensemble models (Random Forest, XGBoost)
* Perform feature selection
* Deploy online (Render / AWS)

---

## Summary

This project demonstrates the transition from isolated models to a complete ML system, focusing on:

* pipeline design
* multi-model integration
* evaluation discipline
* deployment readiness

It reflects practical machine learning beyond textbook implementations.
