import mlflow
import os
import sys

try:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))

    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()

    print(f"Loaded Run ID: {run_id}")

    client = mlflow.tracking.MlflowClient()

    # Get metric history
    metric_history = client.get_metric_history(run_id, "accuracy")

    if not metric_history:
        print(" ERROR: No accuracy metric found in MLflow")
        sys.exit(1)

    accuracy = metric_history[-1].value

    print(f"Final Accuracy: {accuracy}")

    if accuracy < 0.85:
        print(" FAILED: Accuracy below threshold")
        sys.exit(1)
    else:
        print(" PASSED: Accuracy meets threshold")

except Exception as e:
    print(" EXCEPTION OCCURRED:")
    print(str(e))
    sys.exit(1)
