# Use official Python image with Ubuntu
FROM --platform=linux/x86_64 python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN pip install uv

# Create working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY swe_bench_validator/ ./swe_bench_validator/
COPY data_points/ ./data_points/
COPY scripts/ ./scripts/
COPY test_*.py ./

# Install dependencies via UV
RUN uv sync

# Install SWE-bench
RUN pip install swebench>=4.0.4

# Install bash (just in case)
RUN apt-get update && apt-get install -y bash && rm -rf /var/lib/apt/lists/*

# Create default entry point
CMD ["bash"] 