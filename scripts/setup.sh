#!/bin/bash
# LiteServe Setup Script
# Run this first to install all dependencies.

set -e

echo "============================================"
echo "LiteServe Setup"
echo "============================================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python: $PYTHON_VERSION"

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# Core dependencies
echo ""
echo "Installing core dependencies..."
pip install --upgrade pip setuptools wheel -q

pip install \
    torch \
    transformers \
    accelerate \
    sentencepiece \
    protobuf \
    fastapi \
    uvicorn[standard] \
    sse-starlette \
    pyyaml \
    prometheus-client \
    httpx \
    pytest \
    pytest-asyncio \
    -q

# Optional: datasets for perplexity evaluation
echo ""
echo "Installing datasets (for perplexity evaluation)..."
pip install datasets -q || echo "  Warning: datasets install failed (optional)"

# Optional: bitsandbytes for INT8/INT4 quantization (CUDA only)
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
    echo ""
    echo "CUDA detected — installing bitsandbytes..."
    pip install bitsandbytes -q || echo "  Warning: bitsandbytes install failed"
else
    echo ""
    echo "No CUDA — skipping bitsandbytes (quantization requires NVIDIA GPU)"
fi

# Run tests
echo ""
echo "============================================"
echo "Running tests..."
echo "============================================"
PYTHONPATH=. python -m pytest tests/ -v

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Validate: python scripts/validate.py"
echo "  2. Start server: python -m liteserve.server.app --config configs/local.yaml"
echo "  3. Send a request: curl -X POST http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello world\", \"max_tokens\": 50, \"stream\": false}'"
echo "  4. Benchmark: python scripts/benchmark_comparison.py"
echo ""
