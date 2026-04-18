import pickle
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from preprocessing import (
    encode_features,
    get_default_raw_input,
    get_education_num_map,
    get_feature_columns,
    get_input_schema,
    prepare_dataframe,
    scale_features,
    split_data,
)

DATASET_PATH = "dataset.csv"
MODEL_PATH = "system.pkl"


def main():
   
    raw_df = prepare_dataframe(DATASET_PATH)

    input_schema = get_input_schema(raw_df)
    default_raw_input = get_default_raw_input(raw_df)
    education_num_map = get_education_num_map(raw_df)

    model_df = encode_features(raw_df)
    feature_columns = get_feature_columns(model_df)

    X, y_log, y_cls = split_data(model_df)

  
    X_train, X_test, y_log_train, y_log_test, y_cls_train, y_cls_test = train_test_split(
        X, y_log, y_cls, test_size=0.2, random_state=42
    )

    X_train, mean, std = scale_features(X_train)
    X_test = (X_test - mean) / std


    # Baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_log_train)
    dummy_acc = accuracy_score(y_log_test, dummy.predict(X_test))

    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_log_train)
    log_acc = accuracy_score(y_log_test, log_model.predict(X_test))

    # Decision Tree
    tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_model.fit(X_train, y_cls_train)
    tree_acc = accuracy_score(y_cls_test, tree_model.predict(X_test))

    # Random Forest (for comparison)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_log_train)
    rf_acc = accuracy_score(y_log_test, rf_model.predict(X_test))

    # Extract feature importance
    feature_importance = rf_model.feature_importances_

  
    print("----- MODEL PERFORMANCE -----")
    print(f"Dummy Baseline Accuracy: {dummy_acc:.4f}")
    print(f"Logistic Regression Accuracy: {log_acc:.4f}")
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print(f"Decision Tree Accuracy (class): {tree_acc:.4f}")

    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "logistic_model": log_model,
                "tree_model": tree_model,
                "rf_model": rf_model,
                "mean": mean,
                "std": std,
                "feature_columns": feature_columns,
                "input_schema": input_schema,
                "default_raw_input": default_raw_input,
                "education_num_map": education_num_map,
                "feature_importance": feature_importance,
                "metrics": {
                    "dummy_accuracy": float(dummy_acc),
                    "logistic_accuracy": float(log_acc),
                    "rf_accuracy": float(rf_acc),
                    "tree_accuracy": float(tree_acc),
                },
            },
            f,
        )

    print(f"Saved system to {MODEL_PATH}")


if __name__ == "__main__":
    main()
