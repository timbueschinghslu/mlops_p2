# Base image
FROM python:3.12

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y git

# Copy requirements file and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the training script into the container
COPY train.py train.py

# Set default command to run a single training with values from best hyperparameter of project 1
CMD ["python", "train.py", "--checkpoint_dir", "/app/models", "--lr", "1e-4", "--warmup_steps", "13", "--weight_decay", "0.02", "--train_batch_size", "128"]
