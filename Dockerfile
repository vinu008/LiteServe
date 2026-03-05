FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip3 install --no-cache-dir -e ".[dev]"

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000 9090

# Default command
CMD ["python3", "-m", "liteserve.server.app", "--config", "configs/default.yaml"]
