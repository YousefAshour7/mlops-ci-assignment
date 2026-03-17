FROM pytorch/pytorch:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
COPY mnist_train.csv .

CMD ["python", "train.py"]