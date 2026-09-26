#!/bin/zsh

cd '/Users/zhanghuan/Documents/Codex/2026-09-18/m-a/work/pku_quantllm_homework' || exit 1
clear
.venv/bin/python factor/activate_choice.py
result=$?

echo
if [[ $result -eq 0 ]]; then
    echo 'Choice 激活程序已成功完成。'
else
    echo 'Choice 激活未成功，请保留上面的错误代码。'
fi

read '?按回车键关闭此窗口……'
