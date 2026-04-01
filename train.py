import pickle

import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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
    X, y_log, y_lin, y_cls = split_data(model_df)

    (
        X_train,
        X_test,
        y_log_train,
        y_log_test,
        y_lin_train,
        y_lin_test,
        y_cls_train,
        y_cls_test,
    ) = train_test_split(X, y_log, y_lin, y_cls, test_size=0.2, random_state=42)

    X_train, mean, std = scale_features(X_train)
    X_test = (X_test - mean) / std

    lin_model = LinearRegression()
    lin_model.fit(X_train, y_lin_train)
    lin_pred = lin_model.predict(X_test)
    lin_rmse = np.sqrt(mean_squared_error(y_lin_test, lin_pred))

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_log_train)
    log_pred = log_model.predict(X_test)
    log_acc = accuracy_score(y_log_test, log_pred)

    cls_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    cls_model.fit(X_train, y_cls_train)
    cls_pred = cls_model.predict(X_test)
    cls_acc = accuracy_score(y_cls_test, cls_pred)

    print("--- MODEL METRICS ---")
    print(f"Linear Regression RMSE: {lin_rmse:.4f}")
    print(f"Logistic Regression Accuracy: {log_acc:.4f}")
    print(f"Decision Tree Accuracy: {cls_acc:.4f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "linear_model": lin_model,
                "logistic_model": log_model,
                "classifier_model": cls_model,
                "mean": mean,
                "std": std,
                "feature_columns": feature_columns,
                "input_schema": input_schema,
                "default_raw_input": default_raw_input,
                "education_num_map": education_num_map,
                "metrics": {
                    "linear_rmse": float(lin_rmse),
                    "logistic_accuracy": float(log_acc),
                    "classifier_accuracy": float(cls_acc),
                },
            },
            f,
        )

    print(f"Saved system to {MODEL_PATH}")


if __name__ == "__main__":
    main()
