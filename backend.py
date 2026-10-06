import io
import json
import time  # Global time module tracking wrapper
import warnings
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# Core scikit-learn components
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.exceptions import DataConversionWarning

# Model Frameworks
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# =====================================================================
# ➕ MODIFICATION 1: IMPORT SKLEARN MULTI-LAYER PERCEPTRON (ANN)
# =====================================================================
from sklearn.neural_network import MLPRegressor, MLPClassifier

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

app = FastAPI(title="ML Computational Engine API", version="1.0.0")

# =====================================================================
# 🕒 CENTRALIZED TIMING WRAPPER FUNCTION (RUNS ONCE)
# =====================================================================
def run_with_timer(func, *args, **kwargs):
    """Executes model methods while measuring individual execution runtimes."""
    tic = time.time()
    result = func(*args, **kwargs)
    toc = time.time()
    execution_time = toc - tic
    return result, f"{execution_time:.4f}"

def data_preprocessing(b: pd.DataFrame) -> ColumnTransformer:
    """Automated data cleaning and scaling pipeline factory."""
    all_numerical_features = list(b.select_dtypes(include=['int64', 'float64']).columns.values)
    all_categorical_features = list(b.select_dtypes(include=['object', 'category', 'bool']).columns.values)
    
    num_transform = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
    cat_transform = make_pipeline(SimpleImputer(strategy='constant', fill_value='missing'), OrdinalEncoder())
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_tran', num_transform, all_numerical_features),
            ('cat_tran', cat_transform, all_categorical_features),
        ],
        remainder='passthrough'
    )
    return preprocessor

@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    """Profiles incoming dataset metadata parameters for sidebar summary view."""
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents)) if file.filename.endswith('.xlsx') else pd.read_csv(io.BytesIO(contents))
        return {
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "size": int(df.size),
            "numerical_features": len(df.select_dtypes(include=['int64', 'float64']).columns),
            "categorical_features": len(df.select_dtypes(include=['object', 'category']).columns),
            "null_features": len(df.columns[df.isna().any()]),
            "duplicate_rows": int(df.duplicated().sum())
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file structure: {str(e)}")

@app.post("/evaluate")
async def evaluate_pipeline(
    file: UploadFile = File(...),
    problem_type: str = Form(...),
    target_column: str = Form(""),
    drop_columns: str = Form("[]")
):
    """Evaluates algorithms while measuring global end-to-end program runtime execution time."""
    # ⏱️ START GLOBAL TIMER FOR THE ENTIRE CALCULATION PIPELINE RUN
    global_start_time = time.time()
    
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents)) if file.filename.endswith('.xlsx') else pd.read_csv(io.BytesIO(contents))
        
        try:
            to_drop = json.loads(drop_columns)
            if to_drop and any(to_drop):
                df = df.drop(columns=[c for c in to_drop if c in df.columns], errors='ignore')
        except Exception:
            pass

        # Supervised Pipeline Tracking
        if problem_type in ['R', 'C']:
            if not target_column or target_column not in df.columns:
                raise HTTPException(status_code=400, detail=f"Target column '{target_column}' is missing.")
            
            X = df.drop(columns=[target_column], errors='ignore')
            y = df[target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
            
            y_train_flat = y_train.values.ravel() if isinstance(y_train, (pd.DataFrame, pd.Series)) else y_train
            
            preprocessor = data_preprocessing(X)
            results = {}
            
            if problem_type == 'R':
                # =====================================================================
                # ➕ MODIFICATION 2: ADDED MLPRegressor TO REGRESSION TRACK
                # =====================================================================
                models = [
                    LinearRegression(), 
                    KNeighborsRegressor(), 
                    RandomForestRegressor(random_state=42), 
                    GradientBoostingRegressor(random_state=42), 
                    Ridge(),
                    MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
                ]
                names = ['Lin', 'KNN', 'RAF', 'GBR', 'Rid', 'ANN']
                
                for model, name in zip(models, names):
                    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
                    _, runtime = run_with_timer(pipeline.fit, X_train, y_train_flat)
                    results[name] = {
                        "Acc_Train": f"{pipeline.score(X_train, y_train):.4f}",
                        "Acc_Test_R2": f"{pipeline.score(X_test, y_test):.4f}",
                        "Execution_Time_Sec": runtime
                    }
                    
            elif problem_type == 'C':
                # =====================================================================
                # ➕ MODIFICATION 3: ADDED MLPClassifier TO CLASSIFICATION TRACK
                # =====================================================================
                models = [
                    LogisticRegression(max_iter=1000, random_state=42), 
                    KNeighborsClassifier(), 
                    RandomForestClassifier(random_state=42), 
                    GradientBoostingClassifier(random_state=42), 
                    SVC(random_state=42),
                    MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
                ]
                names = ['LogReg', 'KNN', 'RFC', 'GBC', 'SVC', 'ANN']
                
                for model, name in zip(models, names):
                    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
                    _, runtime = run_with_timer(pipeline.fit, X_train, y_train_flat)
                    results[name] = {
                        "Acc_Train": f"{pipeline.score(X_train, y_train):.4f}",
                        "Acc_Test": f"{pipeline.score(X_test, y_test):.4f}",
                        "Execution_Time_Sec": runtime
                    }
            
            # ⏱️ STOP GLOBAL TIMER & CALCULATE TOTAL PROGRAM OVERALL TIME
            global_end_time = time.time()
            total_elapsed = f"{(global_end_time - global_start_time):.4f}"
            return JSONResponse(content={
                "results": results,
                "total_pipeline_time_sec": total_elapsed
            })
            
        # Unsupervised Clustering Track
        elif problem_type == 'G':
            preprocessor = data_preprocessing(df)
            X_transformed = preprocessor.fit_transform(df)
            k = 3
            models = [KMeans(n_clusters=k, random_state=42, n_init='auto'), AgglomerativeClustering(n_clusters=k)]
            names = ['KMeans', 'Hierarchical_Agglomerative']
            formatted_results = {}
            
            for model, name in zip(models, names):
                labels, runtime = run_with_timer(model.fit_predict, X_transformed)
                formatted_results[name] = {
                    "Silhouette": f"{silhouette_score(X_transformed, labels):.4f}",
                    "Davies_Bouldin": f"{davies_bouldin_score(X_transformed, labels):.4f}",
                    "Calinski_Harabasz": f"{calinski_harabasz_score(X_transformed, labels):.4f}",
                    "Execution_Time_Sec": runtime
                }
            
            # ⏱️ STOP GLOBAL TIMER FOR CLUSTERING
            global_end_time = time.time()
            total_elapsed = f"{(global_end_time - global_start_time):.4f}"
            return JSONResponse(content={
                "results": formatted_results,
                "total_pipeline_time_sec": total_elapsed
            })
        else:
            raise HTTPException(status_code=400, detail="Invalid operational track chosen.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Computational Failure: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)