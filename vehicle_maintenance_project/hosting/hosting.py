from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="vehicle_maintenance_project/deployment",     # the local folder containing your files
    repo_id="bkrishnamukund/Vehicle-Engine-Maintenance-Prediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                        # optional: subfolder path inside the repo
)
