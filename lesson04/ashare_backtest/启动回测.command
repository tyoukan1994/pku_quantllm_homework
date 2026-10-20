#!/bin/zsh
set -e
task_root="${0:A:h}"
cd "$task_root"
if [[ -x "$task_root/.venv/bin/python" ]]; then
    task_python="$task_root/.venv/bin/python"
elif [[ -x "$task_root/../../.venv/bin/python" ]]; then
    task_python="$task_root/../../.venv/bin/python"
else
    print "请先运行：python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt"
    read "?按回车退出"
    exit 1
fi
open "http://127.0.0.1:8501"
exec "$task_python" -m streamlit run "$task_root/app.py" --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
