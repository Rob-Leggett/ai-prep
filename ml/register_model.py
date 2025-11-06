import shutil, os

src = "mlruns/0/<run_id>/artifacts/model"
dst = "models/latest_catboost"
os.makedirs("models", exist_ok=True)
shutil.copytree(src, dst, dirs_exist_ok=True)