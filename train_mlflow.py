import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import mlflow
import mlflow.pytorch
import argparse

# -----------------------
# Hyperparameters
# -----------------------
parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=64)

args = parser.parse_args()

learning_rate = args.learning_rate
epochs = args.epochs
batch_size = args.batch_size

# -----------------------
# MLflow Setup
# -----------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Assignment3_Yousef")

run_name = f"lr={learning_rate}_bs={batch_size}_ep={epochs}"

with mlflow.start_run(run_name=run_name):

    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)

    mlflow.set_tag("student_id", "202200497")

    # -----------------------
    # Dataset
    # -----------------------
    transform = transforms.ToTensor()

    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # -----------------------
    # Model
    # -----------------------
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # -----------------------
    # Training Loop
    # -----------------------
    for epoch in range(epochs):

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

        # MLflow logging
        mlflow.log_metric("loss", avg_loss, step=epoch)
        mlflow.log_metric("accuracy", accuracy, step=epoch)

    # -----------------------
    # Save model
    # -----------------------
    mlflow.pytorch.log_model(model, "model")