import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, r2_score, silhouette_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

# ==========================================================================
# PAGE CONFIGURATION & CUSTOM CSS
# ==========================================================================
st.set_page_config(page_title="ML Studio Pro", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1E88E5; margin-bottom: 1rem;}
    .sub-header {font-size: 1.2rem; color: #555; margin-bottom: 2rem;}
    .metric-card {background-color: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 5px solid #1E88E5;}
    div[data-testid="stMetricValue"] {font-size: 1.5rem;}
</style>
""", unsafe_allow_html=True)

# ==========================================================================
# HELPER FUNCTIONS
# ==========================================================================
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

def get_model_and_params(model_name, task):
    params = {}
    if task == "Classification":
        if model_name == "Random Forest":
            params['n_estimators'] = st.sidebar.slider("Number of Trees", 10, 200, 100)
            params['max_depth'] = st.sidebar.slider("Max Depth", 2, 20, 10)
            params['min_samples_split'] = st.sidebar.slider("Min Samples Split", 2, 10, 2)
            model = RandomForestClassifier(**params, random_state=42)
        elif model_name == "Support Vector Machine (SVM)":
            params['C'] = st.sidebar.number_input("Regularization (C)", 0.1, 10.0, 1.0)
            params['kernel'] = st.sidebar.selectbox("Kernel", ['linear', 'rbf', 'poly'])
            model = SVC(**params, random_state=42)
        elif model_name == "Logistic Regression":
            params['C'] = st.sidebar.number_input("Regularization (C)", 0.1, 10.0, 1.0)
            params['solver'] = st.sidebar.selectbox("Solver", ['lbfgs', 'liblinear'])
            model = LogisticRegression(**params, random_state=42, max_iter=1000)
    elif task == "Regression":
        if model_name == "Random Forest Regressor":
            params['n_estimators'] = st.sidebar.slider("Number of Trees", 10, 200, 100)
            params['max_depth'] = st.sidebar.slider("Max Depth", 2, 20, 10)
            model = RandomForestRegressor(**params, random_state=42)
        elif model_name == "Support Vector Regressor (SVR)":
            params['C'] = st.sidebar.number_input("Regularization (C)", 0.1, 10.0, 1.0)
            params['kernel'] = st.sidebar.selectbox("Kernel", ['linear', 'rbf'])
            model = SVR(**params)
        elif model_name == "Linear Regression":
            st.sidebar.info("Linear Regression has no major hyperparameters to tune.")
            model = LinearRegression()
    elif task == "Clustering":
        if model_name == "K-Means":
            params['n_clusters'] = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3)
            model = KMeans(**params, random_state=42, n_init=10)
            
    return model, params

# ==========================================================================
# MAIN APP UI
# ==========================================================================
st.markdown('<h1 class="main-header">🤖 ML Studio Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload your data, configure hyperparameters, and train production-ready Machine Learning models in seconds.</p>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("⚙️ Workflow")
step = st.sidebar.radio("Select Step", ["1. Upload Data", "2. Preprocess", "3. Configure & Train", "4. Evaluate Results"])

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# ==========================================================================
# STEP 1: UPLOAD DATA
# ==========================================================================
if step == "1. Upload Data":
    st.header("📂 Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.session_state.df = df
        st.success("Data loaded successfully!")
        st.write("### Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

# ==========================================================================
# STEP 2: PREPROCESS
# ==========================================================================
elif step == "2. Preprocess":
    if st.session_state.df is None:
        st.warning("Please upload data in Step 1 first.")
    else:
        st.header("🧹 Data Preprocessing")
        df = st.session_state.df
        
        col1, col2 = st.columns(2)
        with col1:
            task = st.selectbox("Select Machine Learning Task", ["Classification", "Regression", "Clustering"])
            st.session_state.task = task
            
            if task != "Clustering":
                target_col = st.selectbox("Select Target Variable (Y)", df.columns.tolist())
                st.session_state.target_col = target_col
                feature_cols = [col for col in df.columns if col != target_col]
            else:
                feature_cols = st.multiselect("Select Features for Clustering", df.columns.tolist(), default=df.columns.tolist())
                st.session_state.feature_cols = feature_cols

        with col2:
            st.subheader("Handle Missing Values")
            num_impute = st.selectbox("Numerical Columns", ["Mean", "Median", "Drop"])
            cat_impute = st.selectbox("Categorical Columns", ["Mode", "Drop"])

        if st.button("Apply Preprocessing"):
            with st.spinner("Processing data..."):
                # Simple imputation logic
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                if target_col in numeric_cols and task != "Clustering":
                    numeric_cols.remove(target_col)
                if target_col in cat_cols and task != "Clustering":
                    cat_cols.remove(target_col)

                if num_impute == "Mean":
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                elif num_impute == "Median":
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                else:
                    df = df.dropna(subset=numeric_cols)

                if cat_impute == "Mode":
                    for col in cat_cols:
                        df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df = df.dropna(subset=cat_cols)

                # Encode categorical variables
                le_dict = {}
                for col in cat_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    le_dict[col] = le

                st.session_state.df_clean = df
                
                if task != "Clustering":
                    X = df[feature_cols]
                    y = df[target_col]
                    st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                else:
                    st.session_state.X_train = df[feature_cols]
                    st.session_state.X_test = df[feature_cols] # For clustering, we evaluate on the same or split, keeping simple here
                    st.session_state.y_train = None
                    st.session_state.y_test = None

                st.success("Preprocessing complete! Data is ready for modeling.")
                st.dataframe(df.head(), use_container_width=True)

# ==========================================================================
# STEP 3: CONFIGURE & TRAIN
# ==========================================================================
elif step == "3. Configure & Train":
    if st.session_state.get('X_train') is None:
        st.warning("Please complete preprocessing in Step 2 first.")
    else:
        st.header("⚙️ Model Configuration")
        task = st.session_state.task
        
        if task == "Classification":
            model_name = st.sidebar.selectbox("Select Algorithm", ["Random Forest", "Support Vector Machine (SVM)", "Logistic Regression"])
        elif task == "Regression":
            model_name = st.sidebar.selectbox("Select Algorithm", ["Random Forest Regressor", "Support Vector Regressor (SVR)", "Linear Regression"])
        elif task == "Clustering":
            model_name = st.sidebar.selectbox("Select Algorithm", ["K-Means"])

        st.sidebar.markdown("---")
        st.sidebar.subheader("Hyperparameters")
        
        model, params = get_model_and_params(model_name, task)
        
        st.write("### Selected Configuration")
        st.json(params)
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner(f"Training {model_name}..."):
                # Scale data for SVM/Linear models
                if "SVM" in model_name or "Linear" in model_name or "Logistic" in model_name:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(st.session_state.X_train)
                    X_test_scaled = scaler.transform(st.session_state.X_test)
                    model.fit(X_train_scaled, st.session_state.y_train if task != "Clustering" else None)
                    st.session_state.predictions = model.predict(X_test_scaled)
                    st.session_state.X_test_scaled = X_test_scaled
                else:
                    model.fit(st.session_state.X_train, st.session_state.y_train if task != "Clustering" else None)
                    st.session_state.predictions = model.predict(st.session_state.X_test)
                    st.session_state.X_test_scaled = st.session_state.X_test # Keep original for tree-based feature importance
                
                st.session_state.model = model
                st.session_state.model_name = model_name
                st.success("Model trained successfully!")
                st.sidebar.success("Training Complete!")

# ==========================================================================
# STEP 4: EVALUATE RESULTS
# ==========================================================================
elif step == "4. Evaluate Results":
    if st.session_state.get('model') is None:
        st.warning("Please train a model in Step 3 first.")
    else:
        st.header("📊 Model Evaluation")
        model = st.session_state.model
        task = st.session_state.task
        y_test = st.session_state.y_test
        y_pred = st.session_state.predictions
        
        col1, col2, col3, col4 = st.columns(4)
        
        if task == "Classification":
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            col1.metric("Accuracy", f"{acc:.2%}")
            col2.metric("Precision", f"{prec:.2%}")
            col3.metric("Recall", f"{rec:.2%}")
            col4.metric("F1-Score", f"{f1:.2%}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, aspect="auto", color_continuous_scale="Blues", 
                               labels=dict(x="Predicted", y="Actual", color="Count"))
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Feature Importance (if tree-based)
            if "Random Forest" in st.session_state.model_name:
                st.subheader("Feature Importance")
                importances = model.feature_importances_
                features = st.session_state.X_train.columns
                fig_fi = px.bar(x=importances, y=features, orientation='h', labels={'x': 'Importance', 'y': 'Feature'})
                st.plotly_chart(fig_fi, use_container_width=True)

        elif task == "Regression":
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            col1.metric("MSE", f"{mse:.2f}")
            col2.metric("RMSE", f"{rmse:.2f}")
            col3.metric("R² Score", f"{r2:.2f}")
            
            # Actual vs Predicted
            st.subheader("Actual vs Predicted Values")
            df_plot = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            fig_scatter = px.scatter(df_plot, x="Actual", y="Predicted", trendline="ols", 
                                     labels={"Actual": "Actual Values", "Predicted": "Predicted Values"},
                                     opacity=0.6)
            st.plotly_chart(fig_scatter, use_container_width=True)

        elif task == "Clustering":
            score = silhouette_score(st.session_state.X_train, y_pred)
            col1.metric("Silhouette Score", f"{score:.3f}", help="Closer to 1 is better")
            
            # PCA for 2D visualization
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(st.session_state.X_train)
            
            df_pca = pd.DataFrame({"PCA1": X_pca[:, 0], "PCA2": X_pca[:, 1], "Cluster": y_pred.astype(str)})
            fig_cluster = px.scatter(df_pca, x="PCA1", y="PCA2", color="Cluster", title="Clusters Visualization (PCA)")
            st.plotly_chart(fig_cluster, use_container_width=True)