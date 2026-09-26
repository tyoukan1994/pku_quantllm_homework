#!/bin/zsh

set -u

script_dir="${0:A:h}"
cd "$script_dir" || exit 1

if [[ ! -x ".venv/bin/python" ]]; then
  echo "找不到项目虚拟环境：$script_dir/.venv"
  read -r "?按回车退出..."
  exit 1
fi

".venv/bin/python" "factor/download_jqdata.py"
exit_code=$?

echo
if [[ $exit_code -eq 0 ]]; then
  echo "聚宽历史数据下载完成。"
else
  echo "聚宽历史数据下载未完成，请查看上面的错误信息。"
fi
read -r "?按回车退出..."
exit $exit_code
