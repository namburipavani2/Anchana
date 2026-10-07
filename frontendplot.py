import streamlit as st
import pandas as pd
import httpx
import json
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Auto-ML Workspace UI", layout="wide")
st.title("📊 Anchana_1.0")
st.write("Upload clean data structures, configure variable constraints, and evaluate Scikit-Learn models in real time.")

BACKEND_URL = https://anchana.vercel.app

# Initialize persistence caches for computed calculations across dropdown state shifts
if "calculation_results" not in st.session_state:
    st.session_state.calculation_results = None
if "total_program_time" not in st.session_state:
    st.session_state.total_program_time = None
if "active_track_code" not in st.session_state:
    st.session_state.active_track_code = None

uploaded_file = st.sidebar.file_uploader("Upload Target Workspace File (.csv, .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
    
    try:
        with httpx.Client() as client:
            meta_res = client.post(f"{BACKEND_URL}/analyze-file", files=files)
            if meta_res.status_code == 200:
                analysis = meta_res.json()
                columns = analysis["columns"]
                
                st.sidebar.markdown("### 🔍 Dataset Profile Metrics")
                st.sidebar.markdown(f"**Shape Matrix:** `{analysis['shape'][0]}` rows × `{analysis['shape'][1]}` columns")
                st.sidebar.markdown(f"**Total Elements Size:** `{analysis['size']}`")
                st.sidebar.markdown(f"**Numerical Parameters:** `{analysis['numerical_features']}`")
                st.sidebar.markdown(f"**Categorical Parameters:** `{analysis['categorical_features']}`")
                st.sidebar.markdown(f"**Missing Value Columns:** `{analysis['null_features']}`")
                st.sidebar.markdown(f"**Identified Duplicate Rows:** `{analysis['duplicate_rows']}`")
                
                track_selection = st.sidebar.selectbox(
                    "Select Machine Learning Track",
                    options=["Regression (R)", "Classification (C)", "Clustering Unsupervised (G)"]
                )
                track_code = track_selection[-2]
                
                target_variable = ""
                if track_code in ['R', 'C']:
                    target_variable = st.sidebar.selectbox("Choose Target Predictor Component (y)", options=columns)
                    
                drop_selections = st.sidebar.multiselect(
                    "Choose Optional Attributes to Omit",
                    options=[col for col in columns if col != target_variable]
                )
                
                if st.sidebar.button("⚙️ Execute Model Pipeline Computations", type="primary"):
                    payload_data = {
                        "problem_type": track_code,
                        "target_column": target_variable,
                        "drop_columns": json.dumps(drop_selections)
                    }
                    active_file = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}
                    
                    with st.spinner("Training processing workflows across isolated backend instances..."):
                        with httpx.Client(timeout=120.0) as client:
                            eval_res = client.post(f"{BACKEND_URL}/evaluate", data=payload_data, files=active_file)
                            if eval_res.status_code == 200:
                                response_json = eval_res.json()
                                st.session_state.calculation_results = response_json["results"]
                                st.session_state.total_program_time = response_json["total_pipeline_time_sec"]
                                st.session_state.active_track_code = track_code
                                st.success("Execution completed successfully!")
                            else:
                                st.error(f"Backend Node Processing Error: {eval_res.text}")

                # =====================================================================
                # ➕ MODIFICATION: PERSISTENT RENDER BLOCKS FOR ISOLATED/GLOBAL PLOTS
                # =====================================================================
                if st.session_state.calculation_results is not None:
                    metric_matrix_df = pd.DataFrame(st.session_state.calculation_results)
                    t_code = st.session_state.active_track_code
                    
                    st.subheader("⚡ Live Computational Results")
                    col1, _ = st.columns(2)
                    with col1:
                        st.metric(label="⌛ Overall Execution Runtime", value=f"{st.session_state.total_program_time} Sec")
                        
                    st.write("#### 📊 Evaluation Score & Runtime Matrix")
                    st.dataframe(metric_matrix_df, use_container_width=True)
                    
                    # --- INDIVIDUAL ISOLATED MODEL ANALYSIS BLOCK ---
                    st.write("#### 🎯 Isolated Model Performance Analysis")
                    available_models = list(metric_matrix_df.columns)
                    selected_model = st.selectbox("Select a model to view individually", options=available_models)
                    
                    single_model_series = metric_matrix_df[selected_model].astype(float)
                    fig_single, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5))
                    
                    scores_series = single_model_series.drop(index="Execution_Time_Sec", errors="ignore")
                    sns.barplot(x=scores_series.index, y=scores_series.values, ax=ax1, palette="Blues_r")
                    ax1.set_title(f"{selected_model} Metric Benchmarks")
                    if t_code in ['R', 'C']:
                        ax1.set_ylim(0, 1.05)
                    ax1.set_ylabel("Index Value")
                    
                    speed_val = single_model_series.loc["Execution_Time_Sec"]
                    ax2.bar(["Execution Time (Sec)"], [speed_val], color="#ff4b4b", width=0.3)
                    ax2.set_title(f"{selected_model} Model Compute Latency")
                    ax2.set_ylabel("Seconds")
                    plt.tight_layout()
                    st.pyplot(fig_single)
                    
                    # --- GLOBAL COMPARISON BLOCKS ---
                    chart_df = metric_matrix_df.drop(index="Execution_Time_Sec", errors="ignore")
                    st.write("#### 📈 Global Performance Distribution Visualisation")
                    fig_global, ax_glob = plt.subplots(figsize=(10, 4))
                    
                    if t_code in ['R', 'C']:
                        melted_df = chart_df.reset_index().melt(id_vars='index')
                        melted_df.columns = ['Evaluation Split', 'Model Framework', 'Accuracy Score Metric']
                        melted_df['Accuracy Score Metric'] = melted_df['Accuracy Score Metric'].astype(float)
                        sns.barplot(data=melted_df, x='Model Framework', y='Accuracy Score Metric', hue='Evaluation Split', ax=ax_glob, palette="Blues_d")
                        ax_glob.set_ylim(0, 1.05)
                        ax_glob.set_title("Cross Model Performance Splits Overview (Excluding Runtimes)")
                    else:
                        metric_matrix_df_clean = chart_df.reset_index()
                        melted_df = metric_matrix_df_clean.melt(id_vars='index')
                        melted_df.columns = ['Validation Metric', 'Model Framework', 'Scoring Index Value']
                        melted_df['Scoring Index Value'] = melted_df['Scoring Index Value'].astype(float)
                        sns.barplot(data=melted_df, x='Validation Metric', y='Scoring Index Value', hue='Model Framework', ax=ax_glob, palette="viridis")
                        ax_glob.set_title("Unsupervised Validation Clusters Variance Profile Comparison")
                        
                    plt.tight_layout()
                    st.pyplot(fig_global)
                    
                    st.write("#### ⏱️ Global Model Speed Analysis")
                    times_df = metric_matrix_df.loc[["Execution_Time_Sec"]].astype(float)
                    st.bar_chart(times_df.T)
            else:
                st.error("Failed to parse communication protocols from the target backend node engine environment.")
    except Exception as network_err:
        st.error(f"Failed to communicate with API server instance: {str(network_err)}")
else:
    st.info("💡 Drop an Excel or CSV file into the workspace initialization panel configuration frame on the left to begin.")
