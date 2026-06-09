# Use a Python 3.10 slim image as the base
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and the model
COPY . .

# Install NLTK data (required by your evaluation script, might be needed by the model API too)
# Run this command inside the container during build
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Expose the port your app runs on inside the container
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]