from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo, login
import os
from google.colab import userdata

# Your Hugging Face token created from access keys in write mode
access_key = userdata.get("HF_TOKEN")
# Login to Hugging Face platform with the access token
login(token=access_key)

# Initialize the API
api = HfApi()

repo_id = "bkrishnamukund/Vehicle-Engine-Maintenance-Prediction"
repo_type = "dataset"
print(access_key)
# Initialize API client
#api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="vehicle_maintenance_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
