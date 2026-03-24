import mlflow
import os
import sys

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("Default")

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)

accuracy = run.data.metrics.get("accuracy", 0)

print(f"Run ID: {run_id}")
print(f"Accuracy: {accuracy}")

if accuracy < 0.85:
    print("❌ Accuracy below threshold")
    sys.exit(1)
else:
    print("✅ Accuracy passed")