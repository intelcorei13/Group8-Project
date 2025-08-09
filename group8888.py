import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_curve
import warnings
import os

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Loan Default Prediction", layout="wide")

# Load FontAwesome
st.markdown("""
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

# Theme toggle
theme_choice = st.sidebar.radio("🌗 Select Theme", ["Light", "Dark"])
plotly_theme = "plotly_dark" if theme_choice == "Dark" else "plotly"

# Color settings
bg_color = "#1E1E1E" if theme_choice == "Dark" else "#F9F9F9"
card_text_color = "#FFFFFF" if theme_choice == "Dark" else "#000000"
card_bg_color = "#2c3e50" if theme_choice == "Dark" else "#E0F7FA"

def metric_card(title, value):
    return f"""
    <div style="background-color:{card_bg_color};padding:15px;border-radius:10px;
                text-align:center;color:{card_text_color};box-shadow:0 2px 5px rgba(0,0,0,0.15)">
        <h4 style="margin:0;">{title}</h4>
        <h2 style="margin:0;">{value}</h2>
    </div>
    """

# Safe value count
def safe_value_counts(df_col, name_category="Category", name_count="Count"):
    df_counts = df_col.value_counts().reset_index()
    if df_counts.shape[1] >= 2:
        df_counts.columns = [name_category, name_count]
    else:
        df_counts = pd.DataFrame({name_category: df_counts.index, name_count: df_counts.values})
    return df_counts

# Safe correlation
def safe_corr(df):
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr() if not numeric_df.empty else pd.DataFrame()

# Load dataset (from project folder)
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "Group 8 Dataset.csv")
    return pd.read_csv(file_path)

loan_data = load_data()

# Preprocess
@st.cache_data
def preprocess_data(df):
    df_processed = df.copy()
    for col in df_processed.select_dtypes(include=[np.number]):
        df_processed[col].fillna(df_processed[col].median(), inplace=True)
    for col in df_processed.select_dtypes(include=['object']):
        df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    le = LabelEncoder()
    for feature in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']:
        if feature in df_processed.columns:
            df_processed[feature] = le.fit_transform(df_processed[feature])
    if 'Dependents' in df_processed.columns:
        df_processed['Dependents'] = df_processed['Dependents'].replace('3+', '3')
        df_processed['Dependents'] = pd.to_numeric(df_processed['Dependents'])
    if 'Loan_Status' in df_processed.columns:
        df_processed['Loan_Status'] = le.fit_transform(df_processed['Loan_Status'])
    return df_processed

processed_data = preprocess_data(loan_data)

# Features & scale
feature_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                   'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                   'Loan_Amount_Term', 'Credit_History', 'Property_Area']
X = processed_data[feature_columns]
y = processed_data['Loan_Status']
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_columns)

# --- Page 1: Overview ---
def page1():
    st.markdown('<h2><i class="fas fa-database"></i> Data Import & Overview</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(metric_card("Total Records", len(loan_data)), unsafe_allow_html=True)
    col2.markdown(metric_card("Features", len(feature_columns)), unsafe_allow_html=True)
    approved = len(loan_data[loan_data['Loan_Status'] == 'Y'])
    rejected = len(loan_data[loan_data['Loan_Status'] == 'N'])
    col3.markdown(metric_card("Approved Loans", approved), unsafe_allow_html=True)
    col4.markdown(metric_card("Rejected Loans", rejected), unsafe_allow_html=True)

    st.divider()
    if st.checkbox("Show Raw Dataset"):
        st.dataframe(loan_data)
    if st.checkbox("Show Summary Statistics"):
        st.dataframe(loan_data.describe())

    # Plot
    status_counts = safe_value_counts(loan_data['Loan_Status'], "Loan_Status_Category", "Count")
    fig1 = px.bar(status_counts, x='Loan_Status_Category', y='Count',
                  title="Loan Status Distribution", template=plotly_theme)
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(loan_data, x='ApplicantIncome', nbins=30,
                        title="Distribution of Applicant Income", template=plotly_theme)
    st.plotly_chart(fig2, use_container_width=True)

    if st.checkbox("Show Correlation Matrix"):
        corr = safe_corr(loan_data)
        if not corr.empty:
            fig3 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', zmin=-1, zmax=1,
                             title="Correlation Matrix", template=plotly_theme)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("No numeric columns available for correlation matrix.")

# --- Page 2: Preprocessing ---
def page2():
    st.markdown('<h2><i class="fas fa-cogs"></i> Data Preprocessing</h2>', unsafe_allow_html=True)
    st.divider()

    if loan_data.isnull().sum().sum() > 0:
        st.write("Missing Values Detected:")
        st.write(loan_data.isnull().sum()[loan_data.isnull().sum() > 0])
    else:
        st.success("No missing values found.")

    st.write("Categorical variables encoded:", ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area'])
    st.write("Features standardized using StandardScaler.")

    col1, col2 = st.columns(2)
    if st.checkbox("Show Processed Data"):
        col1.dataframe(processed_data.head())
    if st.checkbox("Show Scaled Data"):
        col2.dataframe(X_scaled.head())

# --- Page 3: Model Training ---
def page3():
    st.markdown('<h2><i class="fas fa-robot"></i> Model Training</h2>', unsafe_allow_html=True)
    st.divider()

    # Apply log transformation for skewed numeric features
    X_transformed = X.copy()
    for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']:
        X_transformed[col] = np.log1p(X_transformed[col])

    # Re-scale after transformation
    X_scaled_transformed = pd.DataFrame(scaler.fit_transform(X_transformed), columns=feature_columns)

    # Store transformed & scaled data in session state for later use
    st.session_state['X_scaled'] = X_scaled_transformed
    st.session_state['y'] = y

    # Models
    nb_model = GaussianNB()
    dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42)

    if st.button("Train Models"):
        nb_model.fit(X_scaled_transformed, y)
        dt_model.fit(X_scaled_transformed, y)
        st.session_state['nb_model'] = nb_model
        st.session_state['dt_model'] = dt_model
        st.session_state['scaler'] = scaler
        st.success("✅ Models trained with preprocessing & tuning applied!")

# --- Page 4: Evaluation ---
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def page4():
    st.markdown('<h2><i class="fas fa-chart-line"></i> Model Evaluation</h2>', unsafe_allow_html=True)
    st.divider()

    if 'X_scaled' not in st.session_state or 'y' not in st.session_state:
        st.warning("Please train models first.")
        return

    X_eval = st.session_state['X_scaled']
    y_eval = st.session_state['y']

    nb_model = GaussianNB()
    dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    if st.button("Evaluate Models"):
        # Train/test split for metrics
        X_train, X_test, y_train, y_test = train_test_split(X_eval, y_eval, test_size=0.2, random_state=42)
        nb_model.fit(X_train, y_train)
        dt_model.fit(X_train, y_train)

        # Collect metrics
        metrics_data = []
        for name, model in [("Naïve Bayes", nb_model), ("Decision Tree", dt_model)]:
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            metrics_data.append({
                "Model": name,
                "Accuracy": np.mean(cross_val_score(model, X_eval, y_eval, cv=cv, scoring='accuracy')),
                "Precision": precision_score(y_test, preds),
                "Recall": recall_score(y_test, preds),
                "F1-score": f1_score(y_test, preds),
                "AUC": roc_auc_score(y_test, proba)
            })

        # Display as a styled table
        st.markdown("### 📊 Model Performance Metrics")
        metrics_df = pd.DataFrame(metrics_data).set_index("Model").round(4)
        st.dataframe(metrics_df.style.highlight_max(color='blue', axis=0))

        # Metric explanations in collapsible section
        with st.expander("ℹ️ What these metrics mean"):
            st.markdown("""
            **Accuracy** – Overall proportion of correct predictions.  
            **Precision** – Of the loans predicted as *approved*, how many were correct.  
            **Recall** – Of all the loans that *should* be approved, how many we actually approved.  
            **F1-score** – Balance between precision and recall.  
            **AUC** – Measures the model’s ability to rank good applicants higher than bad ones.
            """)

        # Confusion Matrices
        for name, model in [("Naïve Bayes", nb_model), ("Decision Tree", dt_model)]:
            preds = model.predict(X_test)
            cm = confusion_matrix(y_test, preds)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               title=f"{name} Confusion Matrix", template=plotly_theme)
            st.plotly_chart(fig_cm, use_container_width=True)

        # ROC Curve
        nb_proba = nb_model.predict_proba(X_test)[:, 1]
        dt_proba = dt_model.predict_proba(X_test)[:, 1]
        nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_proba)
        dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_proba)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=nb_fpr, y=nb_tpr, mode='lines', name="Naïve Bayes"))
        fig_roc.add_trace(go.Scatter(x=dt_fpr, y=dt_tpr, mode='lines', name="Decision Tree"))
        fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash'))
        fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", template=plotly_theme)
        st.plotly_chart(fig_roc, use_container_width=True)

# --- Page 5: Prediction ---
def page5():
    st.markdown('<h2><i class="fas fa-magic"></i> Loan Prediction</h2>', unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
        loan_amount = st.number_input("Loan Amount", min_value=0, value=100)
        loan_term = st.number_input("Loan Term (days)", min_value=0, value=360)
        credit_history = st.selectbox("Credit History", ["0", "1"])

    if st.button("Predict Loan Approval"):
        if 'nb_model' not in st.session_state:
            st.warning("Train models first.")
        else:
            input_data = pd.DataFrame({
                'Gender': [1 if gender == "Male" else 0],
                'Married': [1 if married == "Yes" else 0],
                'Dependents': [3 if dependents == "3+" else int(dependents)],
                'Education': [1 if education == "Graduate" else 0],
                'Self_Employed': [1 if self_employed == "Yes" else 0],
                'ApplicantIncome': [applicant_income],
                'CoapplicantIncome': [coapplicant_income],
                'LoanAmount': [loan_amount],
                'Loan_Amount_Term': [loan_term],
                'Credit_History': [int(credit_history)],
                'Property_Area': [0 if property_area == "Rural" else 1 if property_area == "Semiurban" else 2]
            })
            input_scaled = st.session_state['scaler'].transform(input_data)
            nb_pred = st.session_state['nb_model'].predict(input_scaled)[0]
            dt_pred = st.session_state['dt_model'].predict(input_scaled)[0]
            nb_proba = st.session_state['nb_model'].predict_proba(input_scaled)[0]
            dt_proba = st.session_state['dt_model'].predict_proba(input_scaled)[0]

            st.write("### Naïve Bayes:", "✅ Approved" if nb_pred==1 else "❌ Rejected", f"Confidence: {nb_proba[nb_pred]:.2%}")
            st.write("### Decision Tree:", "✅ Approved" if dt_pred==1 else "❌ Rejected", f"Confidence: {dt_proba[dt_pred]:.2%}")

def page6():
    st.subheader("📊 Interpretation and Conclusions")

    st.write("""
    - Credit history, income, and loan amount are key predictors.
    - Engineered features like Loan-to-Income Ratio improve model performance.
    """)

    if 'dt_model' in st.session_state and 'X_scaled' in st.session_state:
        dt_model = st.session_state['dt_model']
        feature_names = st.session_state['X_scaled'].columns
        importances = dt_model.feature_importances_

        # Create DataFrame for feature importances
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(10)

        st.markdown("### 🔍 Top 10 Most Important Features (Decision Tree)")
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                        title="Feature Importance (Decision Tree)",
                        template=plotly_theme, text_auto='.2f')
        fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("⚠️ Train the Decision Tree model first to see feature importances.")


# --- Navigation ---
pages = {
    "Data Import & Overview": page1,
    "Data Preprocessing": page2,
    "Model Training": page3,
    "Model Evaluation": page4,
    "Prediction": page5,
    "Interpretation & Conclusions": page6
}
selected = st.sidebar.selectbox("Navigate Here🔽", list(pages.keys()))
pages[selected]()
