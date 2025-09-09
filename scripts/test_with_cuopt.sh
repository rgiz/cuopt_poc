#!/bin/bash
set -euo pipefail

echo "🚀 Starting cuOpt integration test..."

# Start cuOpt if not running
if ! docker ps | grep -q cuopt; then
    echo "📦 Starting cuOpt container..."
    docker-compose up -d cuopt
    
    # Wait for cuOpt to be ready
    echo "⏳ Waiting for cuOpt to be ready..."
    for i in {1..60}; do
        if curl -sf http://localhost:5000/health >/dev/null 2>&1 || \
           curl -sf http://localhost:5000 >/dev/null 2>&1; then
            echo "✅ cuOpt is ready!"
            break
        fi
        echo "   Attempt $i/60..."
        sleep 2
    done
fi

# Set test environment
export TEST_CUOPT_URL="http://localhost:5000"
export CUOPT_URL="http://localhost:5000"

# Run the integration test
echo "🧪 Running cuOpt integration tests..."
pytest tests/integration/test_cuopt_integration.py::TestCuOptIntegration::test_cuopt_end_to_end_with_real_server -v -s

echo "🎉 cuOpt integration test completed!"