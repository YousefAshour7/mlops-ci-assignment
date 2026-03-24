import mlflow
import os
import sys

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()

# Get full run data
run = client.get_run(run_id)

# 🔥 Get ALL metric history
metric_history = client.get_metric_history(run_id, "accuracy")

if not metric_history:
    print("❌ No accuracy metric found")
    sys.exit(1)

# Take last logged value
accuracy = metric_history[-1].value

print(f"Run ID: {run_id}")
print(f"Final Accuracy: {accuracy}")

# Threshold check
if accuracy < 0.85:
    print("❌ Accuracy below threshold")
    sys.exit(1)
else:
    print("✅ Accuracy passed")