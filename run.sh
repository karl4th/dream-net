#!/bin/bash
# DREAM-Net Experiment Runner
# ===========================
# Usage: ./run.sh <experiment.py> [args]
#
# Examples:
#   ./run.sh                                    # Show this help
#   ./run.sh experiments/speaker_switch.py      # Run speaker switch experiment
#   ./run.sh experiments/stress_test.py         # Run stress test
#   ./run.sh experiments/rank_ablation.py       # Run rank ablation
#   ./run.sh experiments/long_cycle.py          # Run long cycle test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Add src to Python path so dream_net can be imported
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

# Activate virtual environment if it exists
if [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# Show help if no arguments
if [ -z "$1" ]; then
    echo "DREAM-Net Experiment Runner"
    echo "==========================="
    echo ""
    echo "Usage: ./run.sh <experiment.py> [args]"
    echo ""
    echo "Available experiments:"
    echo "  experiments/speaker_switch.py   — Single speaker switch test"
    echo "  experiments/stress_test.py      — Multi-speaker stress test"
    echo "  experiments/rank_ablation.py    — Rank ablation study"
    echo "  experiments/long_cycle.py       — Long-cycle memory test"
    echo ""
    echo "Examples:"
    echo "  ./run.sh experiments/speaker_switch.py"
    echo "  ./run.sh experiments/stress_test.py --some-arg"
    exit 0
fi

# Run the experiment
echo "Running: $1"
echo "----------------------------------------"
.venv/bin/python "$@"
