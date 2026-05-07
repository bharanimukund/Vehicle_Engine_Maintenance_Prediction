# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
from vehicle_maintenance_project.model_building.feature_engineering import (
    add_engine_features
)


# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/bkrishnamukund/Vehicle-Engine-Maintenance-Prediction/engine_data.csv"
df_orig = pd.read_csv(DATASET_PATH)
print("Engine Dataset loaded successfully.")

df_fe = add_engine_features(df_orig)
print("Engine new features added successfully.")

# -------------------- Target Column --------------------
target_col = "Engine Condition"

# Split into X (features) and y (target)
X = df_fe.drop(columns=[target_col])
y = df_fe[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]
'''
for f in files:
    api.delete_file(
        path_in_repo=f,
        repo_id="bkrishnamukund/Vehicle-Engine-Maintenance-Prediction",
        repo_type="dataset"
    )
    print(f"Deleted: {f}")
'''
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="bkrishnamukund/Vehicle-Engine-Maintenance-Prediction",
        repo_type="dataset",
        commit_message=f"Upload {file_path}"
    )
