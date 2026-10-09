#!/bin/zsh
set -e
sim_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$sim_dir"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python server.py --port 8765 "$@"
