# Income Intelligence System

A decision-focused machine learning system that predicts income category using probabilistic modeling and multi-model validation.

---

## Overview

This system takes demographic and employment inputs and produces a **final decision**, not just model outputs.

Instead of returning raw predictions, it combines multiple models with a decision layer to generate:

* A clear income decision
* Confidence level
* Model consistency signal

---

## Problem

Given structured data (age, education, work type, etc.), determine:

* Is the individual likely to earn more than 50K?
* How confident is the prediction?
* Are multiple models in agreement?

Traditional ML pipelines output predictions but lack **decision logic** and **consistency validation**, making them unreliable in real-world scenarios.

---

## System Architecture

```
User Input
   ↓
Validation Layer
   ↓
Preprocessing (encoding + scaling)
   ↓
Model Layer
   ├── Logistic Regression (probability)
   ├── Decision Tree (class)
   └── Random Forest (validation)
   ↓
Decision Layer
   ↓
Final Output
```

---

## Models Used

| Model               | Purpose                                              |
| ------------------- | ---------------------------------------------------- |
| Logistic Regression | Provides stable probability estimation               |
| Decision Tree       | Offers interpretable class-based decisions           |
| Random Forest       | Acts as a validation model to detect inconsistencies |

---

## Decision Logic

The system does not directly expose model outputs. Instead, it applies rules:

* Probability > 0.7 → **High Income (High confidence)**
* Probability < 0.3 → **Low Income (High confidence)**
* Otherwise → fallback to Decision Tree classification

### Consistency Check

The system flags conflicts when:

* Logistic prediction and Random Forest prediction disagree

This ensures the output is not blindly trusted.

---

## Model Performance

```
Dummy Baseline Accuracy: ~75.8%
Logistic Regression:     ~84.9%
Random Forest:           ~85.3%
Decision Tree (class):   ~87.8%
```

* Logistic regression is retained for **probability stability**
* Random Forest shows slightly higher accuracy but is used for validation rather than decision

---

## Key Engineering Decisions

### 1. Removed Synthetic Targets

Initial regression-based income estimation was removed to avoid:

* Data leakage
* Unrealistic predictions
* Circular dependencies

### 2. Unified Preprocessing Pipeline

Ensured:

* Same encoding during training and inference
* Same feature ordering
* Same scaling (mean, std)

This prevents training–inference mismatch.

### 3. Multi-Model Validation

Instead of relying on a single model:

* Models are used for **different roles**
* Outputs are validated before forming a decision

---

## Limitations

* Decision thresholds (0.7 / 0.3) are heuristic and not optimized
* Multi-class labels are derived, not real-world income bands
* No extensive hyperparameter tuning
* No model calibration applied

---

## Future Improvements

* Optimize thresholds using ROC curve analysis
* Add feature importance and interpretability tools
* Introduce model calibration for better probability estimates
* Deploy as a production API with monitoring and logging

---

## Tech Stack

* Python
* NumPy, Pandas
* Scikit-learn
* Streamlit

---

## How to Run

### 1. Install dependencies

```
pip install numpy pandas scikit-learn streamlit
```

### 2. Train the model

```
python train.py
```

### 3. Run the application

```
streamlit run app.py
```

---

## Example Output

```json
{
  "decision": "High Income",
  "confidence": "High confidence",
  "probability": 0.82,
  "consistency": "Consistent"
}
```

---

## Summary

This project focuses on **decision-oriented machine learning**, not just prediction.

It demonstrates:

* Pipeline consistency
* Model validation
* Decision-layer design
* Practical deployment thinking
