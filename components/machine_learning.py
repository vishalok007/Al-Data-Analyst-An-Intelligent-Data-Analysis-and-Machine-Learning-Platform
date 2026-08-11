import html
import re
import time
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.automl import (
    automl_regression,
    automl_classification,
    interpret_r2
)
from utils.ai_analyst import explain_automl_results
from utils.machine_learning import (
    prepare_data,
    train_regression,
    train_logistic,
    train_random_forest,
    train_decision_tree_classifier,
    train_decision_tree_regression
)
from utils.chart_style import apply_chart_style
def get_regression_targets(df):
    return df.select_dtypes(include="number").columns.tolist()
def get_classification_targets(df, max_unique=20):
    valid_targets = []

    for col in df.columns:
        unique_count = df[col].nunique(dropna=True)
        if 2 <= unique_count <= max_unique:
            valid_targets.append(col)

    return valid_targets
def render_prediction_download(prediction_df, file_name="predictions.csv"):
    csv = prediction_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Predictions",
        csv,
        file_name,
        "text/csv"
    )
def render_results_download(results_df, file_name):
    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download AutoML Report",
        csv,
        file_name,
        "text/csv"
    )
def add_rank_column(results_df, score_column):
    results = results_df.sort_values(score_column, ascending=False).reset_index(drop=True)

    rank = []
    for i in range(len(results)):
        rank.append(f"{i+1}")

    if "Rank" in results.columns:
        results = results.drop(columns=["Rank"])

    results.insert(0, "Rank", rank)
    return results
def render_regression_results(model_name, y_test, predictions, r2, mae, rmse):
    #st.success(f"{model_name} trained successfully.")
    st.markdown(f"""
    <div class="insight-banner">
       <div>
           <div class="insight-banner-title">Success</div>
           <div class="insight-banner-text">
               {model_name} trained successfully.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-banner">
        <div>
            <div class="insight-banner-title">Regression Summary</div>
            <div class="insight-banner-text">
                Model <strong>{model_name}</strong> completed successfully with
                R² score <strong>{r2:.4f}</strong>.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{r2:.4f}")
    c2.metric("MAE", f"{mae:.4f}")
    c3.metric("RMSE", f"{rmse:.4f}")

    prediction_df = pd.DataFrame({
        "Actual": pd.Series(y_test).reset_index(drop=True),
        "Predicted": pd.Series(predictions).reset_index(drop=True)
    })

    st.markdown("### Prediction Results")
    st.caption("Preview of actual vs predicted values.")
    st.dataframe(prediction_df.head(200), use_container_width=True, height=320)

    fig = px.scatter(
        prediction_df,
        x="Actual",
        y="Predicted",
        title="Actual vs Predicted"
    )
    fig = apply_chart_style(fig)
    fig.update_layout(title_x=0.02)
    st.plotly_chart(fig, use_container_width=True)

    render_prediction_download(prediction_df, "regression_predictions.csv")

    title, message, level = interpret_r2(r2)
    st.markdown("### Model Interpretation")

    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.error(message)
def render_classification_results(model_name, y_test, predictions, accuracy, report, matrix, importance=None):
    st.success(f"{model_name} trained successfully.")

    st.markdown(f"""
    <div class="insight-banner">
        <div>
            <div class="insight-banner-title">Classification Summary</div>
            <div class="insight-banner-text">
                Model <strong>{model_name}</strong> completed successfully with
                accuracy <strong>{accuracy:.4f}</strong>.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy:.4f}")
    c2.metric("Precision", f"{precision:.4f}")
    c3.metric("Recall", f"{recall:.4f}")
    c4.metric("F1 Score", f"{f1:.4f}")

    st.markdown("### Confusion Matrix")
    fig = px.imshow(matrix, text_auto=True, title="Confusion Matrix")
    fig = apply_chart_style(fig)
    fig.update_layout(title_x=0.02)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True, height=320)

    if importance is not None and not importance.empty:
        st.markdown("### Feature Importance")
        st.caption("Top features contributing to the model output.")

        st.dataframe(importance.head(20), use_container_width=True, height=320)

        fig = px.bar(
            importance.head(15).sort_values("Importance", ascending=True),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 Important Features"
        )
        fig = apply_chart_style(fig)
        fig.update_layout(title_x=0.02)
        st.plotly_chart(fig, use_container_width=True)

    prediction_df = pd.DataFrame({
        "Actual": pd.Series(y_test).reset_index(drop=True),
        "Predicted": pd.Series(predictions).reset_index(drop=True)
    })

    st.markdown("### Prediction Results")
    st.caption("Preview of actual vs predicted labels.")
    st.dataframe(prediction_df.head(200), use_container_width=True, height=320)

    render_prediction_download(prediction_df, "classification_predictions.csv")
def render_automl_summary(results, best, problem, execution_time):
    st.markdown("### AutoML Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models Tested", len(results))
    c2.metric("Best Model", best["Model"])

    if problem == "Regression":
        c3.metric("Best R²", f"{best['R² Score']:.4f}")
    else:
        c3.metric("Best Accuracy", f"{best['Accuracy']:.4f}")

    c4.metric("Execution Time", f"{execution_time:.2f}s")
def render_automl_ranking(results, problem):
    score_column = "R² Score" if problem == "Regression" else "Accuracy"
    ranking_title = "Regression Model Comparison" if problem == "Regression" else "Classification Model Comparison"

    ranked_results = add_rank_column(results, score_column)

    st.markdown("### Model Ranking")
    st.dataframe(ranked_results, use_container_width=True, hide_index=True)

    fig = px.bar(
        ranked_results,
        x="Model",
        y=score_column,
        title=ranking_title,
        text=score_column
    )
    fig = apply_chart_style(fig)
    fig.update_layout(title_x=0.02)
    st.plotly_chart(fig, use_container_width=True)

    return ranked_results


def _format_rec_body(body_text):
    lines = [line.strip() for line in body_text.strip().split("\n") if line.strip()]
    html_parts = []
    bullet_items = []

    for line in lines:
        if line.startswith(("o ", "- ", "* ", "• ")):
            cleaned_bullet = re.sub(r"^(?:o|-|\*|•)\s*", "", line).strip()
            bullet_items.append(html.escape(cleaned_bullet))
        else:
            if bullet_items:
                html_parts.append("<ul>" + "".join(f"<li>{b}</li>" for b in bullet_items) + "</ul>")
                bullet_items = []
            html_parts.append(f"<p>{html.escape(line)}</p>")

    if bullet_items:
        html_parts.append("<ul>" + "".join(f"<li>{b}</li>" for b in bullet_items) + "</ul>")

    return "".join(html_parts) or f"<p>{html.escape(body_text)}</p>"


def _parse_recommendations(text):
    raw_items = [item.strip() for item in re.split(r"\n(?=\d+[\.\)]\s+)", text.strip()) if item.strip()]
    items = []
    for raw in raw_items:
        match = re.match(r"^(\d+)[\.\)]\s*(.*?)(?::|\n\s*o|\n\s*-|\n|$)(.*)", raw, re.DOTALL)
        if match:
            num, title, body = match.groups()
            body_clean = body.strip()
            if not body_clean and ":" in raw:
                parts = raw.split(":", 1)
                title = re.sub(r"^\d+[\.\)]\s*", "", parts[0]).strip()
                body_clean = parts[1].strip() if len(parts) > 1 else ""
            items.append((int(num), title.strip(), body_clean))
        else:
            items.append((len(items) + 1, "Action Item", raw))
    return items


def _parse_advisor_sections(report_text):
    raw_sections = re.split(r"\n(?=#{1,3}\s+)", report_text.strip())
    sections = {}
    for sec in raw_sections:
        lines = sec.strip().split("\n")
        if not lines:
            continue
        header_match = re.match(r"^#{1,3}\s+(.*)", lines[0].strip())
        header = header_match.group(1).strip() if header_match else "Overview"
        content = "\n".join(lines[1:]).strip()
        sections[header] = content
    return sections


def generate_local_advisor_report(ranked_results, best_model_name, problem):
    best_score_col = "R² Score" if problem == "Regression" else "Accuracy"
    best_score_val = 0.0
    if ranked_results is not None and not ranked_results.empty:
        best_score_val = float(ranked_results.iloc[0].get(best_score_col, 0.0))

    recs = [
        f"1. Deploy {best_model_name} as Primary Baseline: With the highest cross-validated {best_score_col} of {best_score_val:.4f}, {best_model_name} should serve as the production benchmark for further iteration and deployment.",
        f"2. Cross-Validation & Stability Audit: Execute 5-fold or 10-fold cross-validation across diverse partitions to confirm that the observed {best_score_val:.4f} score generalizes reliably without variance spikes.",
        f"3. Feature Importance & Interpretability: Analyze the relative feature weights and contributions to understand which input signals drive predictions and identify potential redundant attributes.",
        f"4. Hyperparameter Tuning: Apply Bayesian optimization or GridSearchCV to fine-tune learning rate, regularization, or tree depth to push predictive accuracy further.",
        f"5. Data Quality & Feature Engineering: Review missing value handling and consider interaction terms or domain-specific polynomial encodings to extract deeper non-linear patterns."
    ]

    return f"""# Strategic Recommendations
{chr(10).join(recs)}

# Best Model Diagnosis
The automated evaluation pipeline identified **{best_model_name}** as the top-performing algorithm for this {problem.lower()} task. It achieved a benchmark {best_score_col} of **{best_score_val:.4f}**, demonstrating balanced predictive power and architectural stability on the validation holdout.

# Model Comparison
The leaderboard evaluated multiple distinct algorithms. **{best_model_name}** outperformed alternative architectures by minimizing residual error and maximizing sample discriminability across validation splits.

# Performance & Risk Evaluation
- Benchmark Metric ({best_score_col}): **{best_score_val:.4f}**
- Generalization: Risk of extreme overfitting is contained by standard train-test partition validation.
- Recommendation: Perform ongoing monitoring on out-of-distribution batches to verify inference stability.

# Business Impact
This model provides automated decision-support capability, allowing business stakeholders to prioritize actions based on quantified probability and objective performance thresholds.
"""


def render_ai_advisor_report(advisor_report, best_model_name, problem, ranked_results=None):
    is_error = not advisor_report or any(err in advisor_report.lower() for err in ["ai service unavailable", "groq error", "gemini error", "error code: 404"])
    if is_error:
        advisor_report = generate_local_advisor_report(ranked_results, best_model_name, problem)

    st.markdown(f"""
    <div class="advisor-hero">
        <div>
            <div class="advisor-hero-title">AI AutoML Strategic Advisor</div>
            <div class="advisor-hero-subtitle">Comprehensive model audit, diagnostic insights, and prioritized optimization plan for <strong>{best_model_name}</strong>.</div>
        </div>
        <div class="advisor-hero-pill">{problem} Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    sections = _parse_advisor_sections(advisor_report)


    rec_content = ""
    model_diag_content = ""
    comparison_content = ""
    risk_content = ""
    business_content = ""

    for heading, body in sections.items():
        h_lower = heading.lower()
        if any(k in h_lower for k in ["recommendation", "action plan", "strategic", "next step"]):
            rec_content = body
        elif any(k in h_lower for k in ["best model", "diagnosis"]):
            model_diag_content = body
        elif any(k in h_lower for k in ["comparison", "benchmark", "trade-off"]):
            comparison_content = body
        elif any(k in h_lower for k in ["risk", "overfitting", "underfitting", "evaluation"]):
            risk_content = body
        elif any(k in h_lower for k in ["business", "impact", "value"]):
            business_content = body

    if not rec_content and sections:
        if re.search(r"\n\d+[\.\)]\s+", advisor_report):
            rec_content = advisor_report

    tab_titles = [
        "Strategic Action Plan",
        "Best Model Diagnosis",
        "Model Comparison",
        "Performance & Risk",
        "Business Impact",
        "Full Report",
    ]
    tabs = st.tabs(tab_titles)

    # 1. Strategic Action Plan tab
    with tabs[0]:
        st.markdown("### Prioritized Recommendations")
        st.caption("Step-by-step engineering, tuning, and deployment directives.")

        if rec_content:
            rec_items = _parse_recommendations(rec_content)
            if rec_items:
                for num, title, body in rec_items:
                    formatted_body = _format_rec_body(body)
                    st.markdown(f"""
                    <div class="rec-card">
                        <div class="rec-card-header">
                            <span class="rec-badge">{num:02d}</span>
                            <span class="rec-title">{title}</span>
                        </div>
                        <div class="rec-body">
                            {formatted_body}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="advisor-section-card">
                    {rec_content}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Recommendations are available in the Full Report tab.")

    # 2. Best Model Diagnosis tab
    with tabs[1]:
        st.markdown(f"### Top Model: {best_model_name}")
        st.caption("Architectural suitability and strengths.")
        content = model_diag_content or f"The automated pipeline selected **{best_model_name}** as the winning algorithm based on cross-validated performance."
        st.markdown(f"""
        <div class="advisor-section-card">
            {content}
        </div>
        """, unsafe_allow_html=True)

    # 3. Model Comparison tab
    with tabs[2]:
        st.markdown("### Algorithmic Benchmarks")
        st.caption("Trade-offs between complexity, interpretability, and metric score.")
        content = comparison_content or "Detailed comparison metrics are shown in the leaderboard above."
        st.markdown(f"""
        <div class="advisor-section-card">
            {content}
        </div>
        """, unsafe_allow_html=True)

    # 4. Performance & Risk tab
    with tabs[3]:
        st.markdown("### Generalization & Risk Audit")
        st.caption("Overfitting risk, validation fidelity, and potential failure modes.")
        content = risk_content or "Review train vs test score gaps to monitor overfitting risk."
        st.markdown(f"""
        <div class="advisor-section-card">
            {content}
        </div>
        """, unsafe_allow_html=True)

    # 5. Business Impact tab
    with tabs[4]:
        st.markdown("### Practical Business Application")
        st.caption("Decision-making utility and production readiness.")
        content = business_content or "This model is ready for baseline scenario analysis and batch scoring."
        st.markdown(f"""
        <div class="advisor-section-card">
            {content}
        </div>
        """, unsafe_allow_html=True)

    # 6. Full Report tab
    with tabs[5]:
        st.markdown("### Complete Advisory Report")
        st.caption("Raw output from the machine learning intelligence engine.")
        st.markdown(f"""
        <div class="advisor-section-card">
            {advisor_report}
        </div>
        """, unsafe_allow_html=True)


def show_machine_learning(df):

    st.header("Machine Learning Lab")
    st.caption("Train predictive models, compare algorithms, and review automated model recommendations.")
    numeric_targets = get_regression_targets(df)
    classification_targets = get_classification_targets(df)

    st.markdown(f"""
    <div class="insight-banner">
        <div>
            <div class="insight-banner-title">ML Workspace</div>
            <div class="insight-banner-text">
                {len(df):,} rows, {len(df.columns):,} columns,
                {len(numeric_targets)} regression-ready targets, and
                {len(classification_targets)} classification-ready targets detected.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if len(df.columns) > 150:
        st.markdown("""
        <div class="theme-warning-card">
            <div class="theme-warning-title">High-Dimensional Dataset</div>
            <div class="theme-warning-text">
                This dataset contains many columns. Model training and AutoML may take longer than usual.
            </div>
        </div>
        """, unsafe_allow_html=True)
    problem = st.radio(
        "Problem Type",
        ["Classification", "Regression"],
        horizontal=True
    )
    if problem == "Regression":
        if not numeric_targets:
            st.info("No numeric target columns are available for regression.")
            return

        left, right = st.columns(2)
        with left:
            target = st.selectbox("Target Column", numeric_targets)
        with right:
            algorithm = st.selectbox(
                "Regression Algorithm",
                ["Linear Regression", "Decision Tree"]
            )
    else:
        if not classification_targets:
            st.info("No suitable classification target columns were found. Classification targets should usually have a limited number of unique classes.")
            return
        left, right = st.columns(2)
        with left:
            target = st.selectbox("Target Column", classification_targets)
        with right:
            algorithm = st.selectbox(
                "Classification Algorithm",
                ["Logistic Regression", "Random Forest", "Decision Tree"]
            )
    test_size = st.slider(
        "Test Size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5
    )
    b1, b2 = st.columns(2)
    train_clicked = b1.button("Train Model", type="primary", use_container_width=True)
    automl_clicked = b2.button("Run AutoML", use_container_width=True)
    if train_clicked:
        try:
            X_train, X_test, y_train, y_test = prepare_data(
                df,
                target,
                test_size=test_size / 100
            )
            if problem == "Regression":
                if algorithm == "Linear Regression":
                    model_name = "Linear Regression"
                    _, predictions, r2, mae, rmse = train_regression(
                        X_train, X_test, y_train, y_test
                    )
                else:
                    model_name = "Decision Tree Regressor"
                    _, predictions, r2, mae, rmse = train_decision_tree_regression(
                        X_train, X_test, y_train, y_test
                    )
                render_regression_results(model_name, y_test, predictions, r2, mae, rmse)
            else:
                if algorithm == "Logistic Regression":
                    model_name = "Logistic Regression"
                    _, predictions, accuracy, report, matrix = train_logistic(
                        X_train, X_test, y_train, y_test
                    )
                    importance = None
                elif algorithm == "Random Forest":
                    model_name = "Random Forest"
                    _, predictions, accuracy, report, matrix, importance = train_random_forest(
                        X_train, X_test, y_train, y_test
                    )
                else:
                    model_name = "Decision Tree Classifier"
                    _, predictions, accuracy, report, matrix, importance = train_decision_tree_classifier(
                        X_train, X_test, y_train, y_test
                    )
                render_classification_results(
                    model_name,
                    y_test,
                    predictions,
                    accuracy,
                    report,
                    matrix,
                    importance
                )
        except Exception as e:
            st.error(f"Model training failed: {e}")

    if automl_clicked:
        try:
            start = time.time()

            if problem == "Regression":
                results, best = automl_regression(
                    df,
                    target,
                    test_size=test_size / 100
                )
            else:
                results, best = automl_classification(
                    df,
                    target,
                    test_size=test_size / 100
                )
            execution_time = time.time() - start
            best_model_name = best["Model"]
            st.markdown(f"""
            <div class="insight-banner">
               <div>
                    <div class="insight-banner-title">Success</div>
                    <div class="insight-banner-text">
                        AutoML completed successfully. Best model: <strong>{best_model_name}</strong>.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            render_automl_summary(results, best, problem, execution_time)
            ranked_results = render_automl_ranking(results, problem)
            if problem == "Regression":
                render_results_download(ranked_results, "regression_automl_report.csv")
            else:
                render_results_download(ranked_results, "classification_automl_report.csv")
            with st.spinner("Analyzing model performance and synthesizing strategic recommendations..."):
                advisor_report = explain_automl_results(df, ranked_results, problem)
            render_ai_advisor_report(advisor_report, best_model_name, problem, ranked_results=ranked_results)

            if problem == "Regression":
                title, message, level = interpret_r2(best["R² Score"])
                st.markdown("### Best Model Interpretation")
                if level == "success":
                    st.success(message)
                elif level == "warning":
                    st.warning(message)
                else:
                    st.error(message)
        except Exception as e:
            st.error(f"AutoML failed: {e}")
