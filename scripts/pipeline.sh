set -euo pipefail

DATA_DIR="${1:-/data}"
PYTHONPATH=. python3 scripts/run_data_pipeline.py --data-dir "$DATA_DIR"