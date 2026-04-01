import pickle

import numpy as np
import pandas as pd
import streamlit as st

from preprocessing import build_feature_row


MODEL_PATH = "system.pkl"
CLASS_LABELS = {0: "Low", 1: "Medium", 2: "High"}
CLASS_STYLES = {
    "Low": {"bg": "#F5E9DA", "fg": "#7A4A17"},
    "Medium": {"bg": "#E5F1EC", "fg": "#1F6156"},
    "High": {"bg": "#E8EEF7", "fg": "#264A73"},
}


st.set_page_config(
    page_title="Income Intelligence System",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_system():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


def apply_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #f7f4ee;
            --surface: #ffffff;
            --surface-soft: #fbf9f6;
            --line: #e8e1d8;
            --text: #1d2935;
            --muted: #67727d;
            --accent: #24435f;
            --shadow: 0 18px 48px rgba(29, 41, 53, 0.07);
        }

        html, body, [class*="css"] {
            font-family: "Outfit", sans-serif;
        }

        .stApp {
            background: linear-gradient(180deg, #fbfaf7 0%, var(--bg) 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero {
            background: linear-gradient(135deg, #fffdf9 0%, #f6f0e8 100%);
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 2rem;
            box-shadow: var(--shadow);
            margin-bottom: 1.25rem;
        }

        .eyebrow {
            color: var(--accent);
            font-size: 0.76rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            margin-bottom: 0.7rem;
        }

        .hero-title {
            color: var(--text);
            font-size: 2.45rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin: 0;
        }

        .hero-copy {
            color: var(--muted);
            margin-top: 0.6rem;
            max-width: 60ch;
            line-height: 1.65;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 1.2rem;
        }

        .summary-card {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 0.95rem 1rem;
        }

        .summary-label {
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .summary-value {
            color: var(--text);
            font-size: 1.34rem;
            font-weight: 800;
            margin-top: 0.34rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 24px;
            box-shadow: var(--shadow);
            padding: 0.2rem;
        }

        .section-lead {
            color: var(--muted);
            line-height: 1.6;
            margin-bottom: 0.8rem;
        }

        .mini-title {
            color: var(--text);
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 0.3rem;
            margin-bottom: 0.55rem;
        }

        .result-card {
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem;
            margin-bottom: 0.85rem;
        }

        .result-label {
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.38rem;
        }

        .result-value {
            color: var(--text);
            font-size: 2.05rem;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 0.25rem;
        }

        .result-copy {
            color: var(--muted);
            line-height: 1.55;
            margin: 0;
            font-size: 0.92rem;
        }

        .pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.38rem 0.8rem;
            font-size: 0.84rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }

        .stFormSubmitButton button {
            width: 100%;
            border: 0;
            border-radius: 14px;
            background: linear-gradient(135deg, var(--accent) 0%, #36597a 100%);
            color: #ffffff;
            padding: 0.84rem 1rem;
            font-size: 0.98rem;
            font-weight: 800;
            box-shadow: 0 14px 26px rgba(36, 67, 95, 0.18);
        }

        .stFormSubmitButton button:hover {
            background: linear-gradient(135deg, #1d3448 0%, #2f4e6b 100%);
        }

        .stSlider label, .stNumberInput label, .stSelectbox label {
            color: var(--text) !important;
            font-weight: 600 !important;
        }

        .stAlert {
            border-radius: 16px;
        }

        @media (max-width: 900px) {
            .summary-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_schema_map(input_schema):
    return {field["name"]: field for field in input_schema}


def get_options(schema, field_name):
    return schema[field_name]["options"]


def get_default_index(options, default_value):
    return options.index(default_value) if default_value in options else 0


def build_raw_input(user_input, defaults, education_num_map):
    raw_input = dict(defaults)
    raw_input["age"] = int(user_input["age"])
    raw_input["education"] = user_input["education"]
    raw_input["education-num"] = int(education_num_map[user_input["education"]])
    raw_input["workclass"] = user_input["workclass"]
    raw_input["hours-per-week"] = int(user_input["hours_per_week"])
    raw_input["capital-gain"] = float(user_input["capital_gain"])
    raw_input["capital-loss"] = float(user_input["capital_loss"])
    raw_input["sex"] = user_input["gender"]
    raw_input["occupation"] = user_input["occupation"]
    return raw_input


def preprocess_input(raw_input, system):
    feature_row = build_feature_row(raw_input, system["feature_columns"])
    mean = np.asarray(system["mean"], dtype=np.float64)
    std = np.asarray(system["std"], dtype=np.float64)
    return (feature_row - mean) / std


def run_predictions(processed_input, system):
    probability = float(system["logistic_model"].predict_proba(processed_input)[0][1])
    estimated_income = float(system["linear_model"].predict(processed_input)[0])
    income_class_idx = int(system["classifier_model"].predict(processed_input)[0])
    return {
        "probability": probability,
        "estimated_income": estimated_income,
        "income_class": CLASS_LABELS.get(income_class_idx, "Medium"),
    }


def validate_inputs(user_input):
    if user_input["capital_gain"] < 0 or user_input["capital_loss"] < 0:
        return "Capital gain and capital loss must be zero or greater."
    if not user_input["education"] or not user_input["workclass"] or not user_input["occupation"]:
        return "Please complete the required fields before predicting."
    return None


def render_header(metrics):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">Decision Support Interface</div>
            <h1 class="hero-title">Income Intelligence System</h1>
            <div class="hero-copy">Predict income probability, estimated income, and income class with a cleaner layout and metrics that better reflect the current pipeline.</div>
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="summary-label">Logistic Accuracy</div>
                    <div class="summary-value">{metrics["logistic_accuracy"]:.1%}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-label">Linear RMSE</div>
                    <div class="summary-value">{metrics["linear_rmse"]:.0f}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-label">Tree Accuracy</div>
                    <div class="summary-value">{metrics["classifier_accuracy"]:.1%}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_panel(system):
    schema = get_schema_map(system["input_schema"])
    defaults = system["default_raw_input"]

    with st.container(border=True):
        st.subheader("Profile Builder")
        st.markdown(
            '<div class="section-lead">Build a profile on the left and run the prediction when you are ready. The sections are short and aligned to reduce visual friction.</div>',
            unsafe_allow_html=True,
        )

        with st.form("prediction_form"):
            st.markdown('<div class="mini-title">Identity</div>', unsafe_allow_html=True)
            age = st.slider("Age", 18, 70, int(np.clip(defaults["age"], 18, 70)))
            gender_options = get_options(schema, "sex")
            gender = st.selectbox(
                "Gender",
                gender_options,
                index=get_default_index(gender_options, defaults["sex"]),
            )

            st.markdown('<div class="mini-title">Education And Work</div>', unsafe_allow_html=True)
            education_options = get_options(schema, "education")
            education = st.selectbox(
                "Education level",
                education_options,
                index=get_default_index(education_options, defaults["education"]),
            )
            workclass_options = get_options(schema, "workclass")
            workclass = st.selectbox(
                "Workclass",
                workclass_options,
                index=get_default_index(workclass_options, defaults["workclass"]),
            )
            occupation_options = get_options(schema, "occupation")
            occupation = st.selectbox(
                "Occupation",
                occupation_options,
                index=get_default_index(occupation_options, defaults["occupation"]),
            )
            hours_per_week = st.slider(
                "Hours per week",
                1,
                100,
                int(np.clip(defaults["hours-per-week"], 1, 100)),
            )

            st.markdown('<div class="mini-title">Capital Inputs</div>', unsafe_allow_html=True)
            gain_col, loss_col = st.columns(2)
            with gain_col:
                capital_gain = st.number_input(
                    "Capital gain",
                    min_value=0.0,
                    value=float(max(defaults["capital-gain"], 0.0)),
                    step=100.0,
                )
            with loss_col:
                capital_loss = st.number_input(
                    "Capital loss",
                    min_value=0.0,
                    value=float(max(defaults["capital-loss"], 0.0)),
                    step=100.0,
                )

            predict_clicked = st.form_submit_button("Predict")

    return predict_clicked, {
        "age": age,
        "gender": gender,
        "education": education,
        "workclass": workclass,
        "occupation": occupation,
        "hours_per_week": hours_per_week,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
    }


def render_result_card(label, value, copy_text):
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">{label}</div>
            <div class="result-value">{value}</div>
            <p class="result-copy">{copy_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(predictions):
    probability = predictions["probability"]
    estimated_income = predictions["estimated_income"]
    income_class = predictions["income_class"]
    class_style = CLASS_STYLES[income_class]
    confidence_gap = abs(probability - 0.5)
    confidence = "High confidence" if confidence_gap >= 0.25 else "Moderate confidence" if confidence_gap >= 0.12 else "Low confidence"

    with st.container(border=True):
        st.subheader("Prediction Summary")
        st.markdown(
            '<div class="section-lead">The outputs are ordered from decision to interpretation so the user can understand the result quickly and then review the supporting signals.</div>',
            unsafe_allow_html=True,
        )

        render_result_card(
            "Estimated Income",
            f"${estimated_income:,.0f}",
            "Predicted using the saved linear model with the same preprocessing and feature ordering used during training.",
        )
        render_result_card(
            "Income Probability",
            f"{probability:.1%}",
            f"Probability of income above 50K from logistic regression. {confidence}.",
        )
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Income Class</div>
                <div class="pill" style="background:{class_style["bg"]}; color:{class_style["fg"]};">{income_class}</div>
                <p class="result-copy">Decision tree output grouped into low, medium, and high bands for easier reading.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Probability Gauge")
        st.progress(probability, text=f"Income above 50K: {probability:.1%}")

        threshold_df = pd.DataFrame(
            {"Probability": [probability * 100, 50.0]},
            index=["Prediction", "Threshold"],
        )
        st.markdown("#### Probability vs Threshold")
        st.bar_chart(threshold_df)


def render_empty_state():
    with st.container(border=True):
        st.subheader("Prediction Summary")
        st.markdown(
            '<div class="section-lead">Once you run a prediction, the summary cards and probability chart will appear here in a single aligned review area.</div>',
            unsafe_allow_html=True,
        )
        st.info("No prediction yet. Complete the form and click Predict.")


def main():
    apply_styles()
    system = load_system()
    render_header(system["metrics"])

    left_col, right_col = st.columns([0.96, 1.04], gap="large")

    with left_col:
        predict_clicked, user_input = render_input_panel(system)

    with right_col:
        if predict_clicked:
            validation_error = validate_inputs(user_input)
            if validation_error:
                st.warning(validation_error)
            else:
                raw_input = build_raw_input(
                    user_input,
                    system["default_raw_input"],
                    system["education_num_map"],
                )
                processed_input = preprocess_input(raw_input, system)
                predictions = run_predictions(processed_input, system)
                st.success("Prediction ready.")
                render_results(predictions)
        else:
            render_empty_state()


if __name__ == "__main__":
    main()
