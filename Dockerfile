# SDOH Risk Screening Model - Reproducible Container
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY docs/ ./docs/
COPY models/ ./models/
COPY results/ ./results/
COPY *.py ./
COPY *.md ./

# Create data directory (user will mount their data)
RUN mkdir -p data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create non-root user for security
RUN useradd -m -u 1000 sdoh && chown -R sdoh:sdoh /app
USER sdoh

# Default command
CMD ["python", "-c", "print('SDOH Risk Screening Model Container Ready')"]