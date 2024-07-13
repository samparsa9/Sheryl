# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set environment variables (if needed)
ENV PYTHONUNBUFFERED=1

# Specify the command to run the application
CMD ["python", "src/portfolio_management/JOKR_strat.py"]
