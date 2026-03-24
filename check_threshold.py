import mlflow
import os
import sys

try:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))

    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()

    print(f"Run ID: {run_id}")

    client = mlflow.tracking.MlflowClient()

    history = client.get_metric_history(run_id, "accuracy")

    if not history:
        print("No accuracy found")
        sys.exit(1)

    accuracy = history[-1].value
    print(f"Final Accuracy: {accuracy}")

    if accuracy < 0.85:
        print("BELOW THRESHOLD")
        sys.exit(1)
    else:
        print("ABOVE THRESHOLD")

except Exception as e:
    print("ERROR:", str(e))
    sys.exit(1)