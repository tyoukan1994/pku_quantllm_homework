#!/bin/zsh

set -u
SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR" || exit 1

if [[ ! -x ".venv/bin/python" ]]; then
  echo "没有找到项目虚拟环境：$SCRIPT_DIR/.venv"
  read -r "?按回车键关闭窗口..."
  exit 1
fi

LICENSE_SCRIPT="${HOME}/Downloads/20250911081305_RQData/make.sh"
if [[ -f "$LICENSE_SCRIPT" ]]; then
  ".venv/bin/python" factor/check_rqdata.py --license-script "$LICENSE_SCRIPT"
else
  ".venv/bin/python" factor/check_rqdata.py
fi
exit_code=$?

echo
read -r "?按回车键关闭窗口..."
exit "$exit_code"
