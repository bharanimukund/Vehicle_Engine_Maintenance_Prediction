import pandas as pd
import numpy as np
import sklearn
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

import xgboost as xgb
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_recall_curve,
    average_precision_score
)

import mlflow
import joblib

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Vehicle_Engine_Maintenance_Prediction_MLOps_Experiment_Tracking")

api = HfApi(token=os.getenv("HF_TOKEN"))

Xtrain_path = "https://huggingface.co/datasets/bkrishnamukund/Vehicle-Engine-Maintenance-Prediction/resolve/main/Xtrain.csv"
Xtest_path = "https://huggingface.co/datasets/bkrishnamukund/Vehicle-Engine-Maintenance-Prediction/resolve/main/Xtest.csv"
ytrain_path = "https://huggingface.co/datasets/bkrishnamukund/Vehicle-Engine-Maintenance-Prediction/resolve/main/ytrain.csv"
ytest_path = "https://huggingface.co/datasets/bkrishnamukund/Vehicle-Engine-Maintenance-Prediction/resolve/main/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# -------------------- Class Imbalance --------------------
scale_pos_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print("Scale pos weight:", scale_pos_weight)

# -------------------- Base Model --------------------
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)

model_pipeline = make_pipeline(
    RobustScaler(),
    xgb_model
)

# -------------------- Hyperparameter Search Space --------------------
param_grid = {
    "xgbclassifier__n_estimators": [100, 200, 300],
    "xgbclassifier__max_depth": [2, 3, 4, 5],
    "xgbclassifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "xgbclassifier__subsample": [0.7, 0.85, 1.0],
    "xgbclassifier__colsample_bytree": [0.5, 0.7, 0.9],
    "xgbclassifier__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    "xgbclassifier__gamma": [0, 0.1, 0.2],
    "xgbclassifier__min_child_weight": [1, 3, 5]
}

print("Starting RandomizedSearchCV...")

# -------------------- MLflow --------------------
with mlflow.start_run(run_name="XGB_Vehicle_Maintenance_CV_Optimized"):

    # -------------------- Randomized Search --------------------
    rand_search = RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=param_grid,
        n_iter=50,
        scoring="recall",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    rand_search.fit(Xtrain, ytrain)

    print("RandomizedSearchCV completed.")
    print("Best parameters:", rand_search.best_params_)

    # -------------------- MLflow logging for ALL params --------------------
    results = rand_search.cv_results_

    for i in range(len(results['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("mean_cv_score", results['mean_test_score'][i])
            mlflow.log_metric("std_cv_score", results['std_test_score'][i])

    mlflow.log_params(rand_search.best_params_)

    # -------------------- Best model --------------------
    base_model = clone(rand_search.best_estimator_)

    print("Base model cloned successfully.")

    # -------------------- Calibration (CORRECT WAY) --------------------
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=5
    )

    calibrated_model.fit(Xtrain, ytrain)

    print("Calibration completed.")

    # -------------------- PROBABILITY OUTPUT --------------------
    y_train_proba = calibrated_model.predict_proba(Xtrain)[:, 1]
    y_test_proba = calibrated_model.predict_proba(Xtest)[:, 1]

    # -------------------- SAFE PROBABILITY CHECK --------------------
    print("Train prob stats:",
          np.min(y_train_proba),
          np.mean(y_train_proba),
          np.max(y_train_proba))

    print("Test prob stats:",
          np.min(y_test_proba),
          np.mean(y_test_proba),
          np.max(y_test_proba))

    best_thresh = 0.5

    print("Optimal Threshold (CV-based):", best_thresh)

    mlflow.log_param("classification_threshold", best_thresh)

    # -------------------- Predictions --------------------
    y_pred_train = (y_train_proba >= best_thresh).astype(int)
    y_pred_test = (y_test_proba >= best_thresh).astype(int)

    # -------------------- Reports --------------------
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    print("Train Performance:")
    print(pd.DataFrame(train_report).T)

    print("Test Performance:")
    print(pd.DataFrame(test_report).T)

    # -------------------- MLflow metrics --------------------
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],

        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    tn, fp, fn, tp = confusion_matrix(ytest, y_pred_test).ravel()

    mlflow.log_metrics({
        "test_false_positives": fp,
        "test_false_negatives": fn,
        "test_true_positives": tp,
        "test_true_negatives": tn
    })

    pr_auc = average_precision_score(ytest, y_test_proba)
    mlflow.log_metric("test_pr_auc", pr_auc)

    # -------------------- Model logging --------------------
    mlflow.sklearn.log_model(
        calibrated_model,
        name="xgb_pipeline_best",
        input_example=Xtrain.head(5)
    )

    # Save the model locally
    model_path = "best_Vehicle_Engine_Maintenance_Prediction_model_v1.joblib"
    joblib.dump(calibrated_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "bkrishnamukund/Vehicle-Engine-Maintenance-Prediction"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_Vehicle_Engine_Maintenance_Prediction_model_v1.joblib",
        path_in_repo="best_Vehicle_Engine_Maintenance_Prediction_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type
    )

# ----------------------------
# End the main MLflow run
# ----------------------------
mlflow.end_run()
