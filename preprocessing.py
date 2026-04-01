import numpy as np
import pandas as pd


INCOME_TARGET = "income"
ESTIMATED_INCOME = "Estimated_Income"
INCOME_CLASS = "Income_Class"
MODEL_EXCLUDED_COLUMNS = {INCOME_TARGET, ESTIMATED_INCOME, INCOME_CLASS}


def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def clean_data(df):
    df = df.copy()
    df.replace("?", np.nan, inplace=True)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    return df


def encode_target(df):
    df = df.copy()
    df[INCOME_TARGET] = (
        df[INCOME_TARGET]
        .astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .map({"<=50K": 0, ">50K": 1})
        .astype(int)
    )
    return df


def create_regression_target(df):
    df = df.copy()
    hours = df["hours-per-week"].clip(lower=1)
    capital_gain = df["capital-gain"].clip(lower=0)
    capital_loss = df["capital-loss"].clip(lower=0)

    df[ESTIMATED_INCOME] = (
        np.power(hours, 1.25) * 920
        + df["education-num"] * 2600
        + np.sqrt(capital_gain) * 115
        - np.log1p(capital_loss) * 850
        + df["age"] * 145
    )
    return df


def create_classification_target(df):
    df = df.copy()
    low_cutoff = df[ESTIMATED_INCOME].quantile(0.33)
    high_cutoff = df[ESTIMATED_INCOME].quantile(0.66)

    def classify(value):
        if value <= low_cutoff:
            return 0
        if value <= high_cutoff:
            return 1
        return 2

    df[INCOME_CLASS] = df[ESTIMATED_INCOME].apply(classify)
    return df


def prepare_dataframe(path):
    df = load_data(path)
    df = clean_data(df)
    df = encode_target(df)
    df = create_regression_target(df)
    df = create_classification_target(df)
    return df


def get_input_schema(df):
    feature_df = df.drop(columns=[INCOME_TARGET, ESTIMATED_INCOME, INCOME_CLASS])
    schema = []

    for col in feature_df.columns:
        if feature_df[col].dtype == "object":
            options = sorted(feature_df[col].astype(str).unique().tolist())
            schema.append({"name": col, "type": "categorical", "options": options})
        else:
            schema.append(
                {
                    "name": col,
                    "type": "numeric",
                    "min": float(feature_df[col].min()),
                    "max": float(feature_df[col].max()),
                    "mean": float(feature_df[col].mean()),
                }
            )

    return schema


def get_default_raw_input(df):
    feature_df = df.drop(columns=[INCOME_TARGET, ESTIMATED_INCOME, INCOME_CLASS])
    defaults = {}

    for col in feature_df.columns:
        if feature_df[col].dtype == "object":
            defaults[col] = feature_df[col].mode()[0]
        else:
            defaults[col] = float(feature_df[col].mean())

    return defaults


def get_education_num_map(df):
    mapping = (
        df[["education", "education-num"]]
        .drop_duplicates()
        .sort_values("education")
        .set_index("education")["education-num"]
        .to_dict()
    )
    return {key: int(value) for key, value in mapping.items()}


def encode_features(df):
    return pd.get_dummies(df, drop_first=True, dtype=int)


def get_feature_columns(encoded_df):
    return [col for col in encoded_df.columns if col not in MODEL_EXCLUDED_COLUMNS]


def split_data(df):
    X = df.drop(columns=list(MODEL_EXCLUDED_COLUMNS))

    return (
        X.to_numpy(dtype=np.float64),
        df[INCOME_TARGET].to_numpy(dtype=np.int64),
        df[ESTIMATED_INCOME].to_numpy(dtype=np.float64),
        df[INCOME_CLASS].to_numpy(dtype=np.int64),
    )


def scale_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    return (X - mean) / std, mean, std


def build_feature_row(raw_input, feature_columns):
    encoded = pd.get_dummies(pd.DataFrame([raw_input]), drop_first=True, dtype=int)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    return encoded.to_numpy(dtype=np.float64)
