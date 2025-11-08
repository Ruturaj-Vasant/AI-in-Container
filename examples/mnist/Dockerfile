# Use the official PyTorch image as the base
FROM pytorch/pytorch:latest

# Set working directory
WORKDIR /app

# Copy MNIST example files into container
COPY . /app

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) Add matplotlib for later plotting if needed
RUN pip install matplotlib

# Default command to run training (you can change epochs etc. later)
CMD ["python", "main.py", "--epochs", "1", "--batch-size", "64", "--lr", "0.01"]
