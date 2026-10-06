import streamlit as st
import pandas as pd
import httpx
import json
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Auto-ML Workspace UI", layout="wide")
st.title("📊 Anchana_1.0")
st.write("Upload clean data structures, configure variable constraints, and evaluate Scikit-Learn models in real time.")

# Pointing explicitly to the open server port 8001
BACKEND_URL = "http://127.0.0.1:8002"

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
                
                # =====================================================================
                # ➕ MODIFICATION 4: FIXED BUG WITH ANALYSIS['SHAPE'] RENDER STRING
                # =====================================================================
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
                    st.subheader(f"⚡ Live Computational Results: {track_selection}")
                    
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
                                raw_data = response_json["results"]
                                total_program_time = response_json["total_pipeline_time_sec"]
                                
                                metric_matrix_df = pd.DataFrame(raw_data)
                                st.success("Execution completed successfully!")
                                
                                # 💡 Render the Single Overall Execution Time KPI block scorecard
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric(label="⌛ Overall Execution Runtime", value=f"{total_program_time} Sec")
                                    
                                st.write("#### 📊 Evaluation Score & Runtime Matrix")
                                st.dataframe(metric_matrix_df, use_container_width=True)
                                
                                # Filter clean dataset ONCE here right before graph plotting blocks
                                chart_df = metric_matrix_df.drop(index="Execution_Time_Sec", errors="ignore")
                                
                                st.write("#### 📈 Performance Distribution Visualisation")
                                fig, ax = plt.subplots(figsize=(10, 4))
                                
                                if track_code in ['R', 'C']:
                                    melted_df = chart_df.reset_index().melt(id_vars='index')
                                    melted_df.columns = ['Evaluation Split', 'Model Framework', 'Accuracy Score Metric']
                                    melted_df['Accuracy Score Metric'] = melted_df['Accuracy Score Metric'].astype(float)
                                    sns.barplot(data=melted_df, x='Model Framework', y='Accuracy Score Metric', hue='Evaluation Split', ax=ax, palette="Blues_d")
                                    ax.set_ylim(0, 1.05)
                                    ax.set_title("Cross Model Performance Splits Overview (Excluding Runtimes)")
                                else:
                                    metric_matrix_df_clean = chart_df.reset_index()
                                    melted_df = metric_matrix_df_clean.melt(id_vars='index')
                                    melted_df.columns = ['Validation Metric', 'Model Framework', 'Scoring Index Value']
                                    melted_df['Scoring Index Value'] = melted_df['Scoring Index Value'].astype(float)
                                    sns.barplot(data=melted_df, x='Validation Metric', y='Scoring Index Value', hue='Model Framework', ax=ax, palette="viridis")
                                    ax.set_title("Unsupervised Validation Clusters Variance Profile Comparison")
                                    
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Single independent speed bar-chart block section
                                st.write("#### ⏱️ Model Speed Analysis")
                                times_df = metric_matrix_df.loc[["Execution_Time_Sec"]].astype(float)
                                st.bar_chart(times_df.T)
                            else:
                                st.error(f"Backend Node Processing Error: {eval_res.text}")
            else:
                st.error("Failed to parse communication protocols from the target backend node engine environment.")
    except Exception as network_err:
        st.error(f"Failed to communicate with API server instance: {str(network_err)}")
else:
    st.info("💡 Drop an Excel or CSV file into the workspace initialization panel configuration frame on the left to begin.")