#!/bin/zsh

set -u
PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR" || exit 1

if [[ ! -x ".venv/bin/python" ]]; then
    echo "未找到项目 Python 环境，请先完成 README 中的安装步骤。"
    read "?按回车关闭窗口……"
    exit 1
fi

".venv/bin/python" "ctpdemo/secure_login.py"
exit_code=$?
echo
read "?按回车关闭窗口……"
exit $exit_code
