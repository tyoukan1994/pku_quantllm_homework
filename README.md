# 当代量化交易系统的原理与实现｜作业工作区

本仓库为第二讲课程作业。课程讨论区是 [aslan9/pku_quantllm](https://github.com/aslan9/pku_quantllm)，不是本项目的代码仓库。

## 项目内容

- `raw/`：课堂照片信息的人工转录与外部来源记录。
- `wiki/`：持续维护的作业知识库；从 [索引](wiki/index.md) 开始阅读。
- `ctpdemo/`：基于 `vnpy_ctp` 的最小图形界面程序，包含行情打印和只提示、不自动下单的演示信号。
- `factor/`：RQData因子研究、Alpha158机器学习对照和可复现报告。

## CTP demo 的当前边界

程序启动 VeighNa Trader 自带图形界面。通过“系统 → 连接 CTP”输入 SimNow 仿真账户信息；通过主界面订阅合约；行情会显示在界面并打印到启动程序的终端。自带交易组件可手动下单、撤单和查看订单，具体控件以安装的 vn.py 版本为准。演示信号只在终端打印，不自动发单。

本项目已经在这台 Apple 芯片 Mac 的 `.venv` 独立环境中完成安装和导入验证：Python 3.12.7、vn.py 4.4.0、`vnpy_ctp` 6.7.7.2。`MdApi`、`TdApi` 和 `CtpGateway` 均可载入，`pip check` 未发现依赖冲突。已经使用获授权的 SimNow 账号验证第二套 7×24 柜台登录成功；行情订阅和下单尚未实测。

当前 `vnpy_ctp` 主分支 6.7.11.4 在 Mac 上会因其 C++ 封装与所带 Mac CTP 头文件不匹配而编译失败，因此采用官方发布页标注支持 Mac 编译的 6.7.7.2。源码保留在被 Git 忽略的 `vnpy_ctp_source/`；不要随意移动或删除，后续重装仍会用到。安装过程以 [vn.py 官方 Mac 安装指南](https://www.vnpy.com/docs/cn/community/install/mac_install.html) 和 [`vnpy_ctp` 官方仓库](https://github.com/vnpy/vnpy_ctp)为依据。

安装好后，在本目录运行：

```bash
.venv/bin/python ctpdemo/run.py
```

如果需要手动进入同一环境，先在本目录运行 `source .venv/bin/activate`。项目内的 `.vntrader/` 用于保存本机运行配置，已被 Git 忽略。

连接信息已经在图形界面填写过时，可运行以下安全诊断。程序会使用本机配置连接 20 秒，但不会打印用户名或密码：

```bash
.venv/bin/python ctpdemo/check_connection.py --seconds 20
```

不要把 `.vntrader/connect_ctp.json` 上传或发送给他人；其中包含本机保存的 SimNow 凭证。

启动前可用纯标准库运行本地演示信号测试：

```bash
python3 -m unittest discover -s ctpdemo/tests -v
```

不要把交易账号或密码写进代码、截图或 GitHub。连接账户前先核对 SimNow 所选环境及其开放时间；仅使用仿真账户。

如需临时验证一个已授权账号，可双击项目根目录的 `connect_simnow.command`。它会在本机隐藏输入交易密码，且不保存账号或密码；诊断只做柜台登录，不订阅行情、不发送委托。

要运行只读行情链路，可双击 `subscribe_simnow.command`。程序会隐藏输入凭据，登录后查询合约表，让用户选择一个期货合约，打印最多 10 条 Tick，并将结果保存在被 Git 忽略的 `.vntrader/ticks/`。该脚本没有下单入口。

[SimNow 官方环境说明](https://www.simnow.com.cn/product.action)列出盘中仿真环境与供 API 开发测试的第二套环境。新注册用户使用第二套环境可能需要等到第三个交易日；周末能否登录、是否有行情应按所选环境的服务时间和实际返回的日志判断，不能只看网页注册是否成功。

## 因子研究当前结果

数据源采用米筐RQData，股票池为每个信号月末当时的沪深300和中证500成分股并集。2015—2024年为开发区间，因子、模型和参数冻结后，使用2025年1—12月收益完成一次性最终样本外检验。第一版离散权重报告见 [`factor/results/ff3_combo_report.md`](factor/results/ff3_combo_report.md)。

升级版加入vn.py Alpha158、量价反共振、隔夜—日内分解、Ridge、LightGBM和M3/MPS适配的PyTorch MLP，并按年度扩展窗口生成2020—2024滚动预测。冻结的主候选是三模型等权秩集成。2025年平均Rank IC为0.0256，但95%置信区间包含零，扣费多空收益为-22.47%，因此最终结论是样本外未验证出稳定有效性。完整方法、结果和限制见[最终因子挖掘报告](factor/results/alpha_ml/final_factor_mining_report.md)。授权原始行情和股票级宽表仅保存在被Git忽略的`data/private/`。

复现已有缓存上的模型比较：

```bash
PYTHONPATH=factor .venv/bin/python factor/run_alpha_ml.py
```

在Apple Silicon上，Codex沙箱可能不暴露Metal设备；需要从本机终端运行上述命令时，脚本会自动选择MPS。SimNow/CTP只用于仿真交易和实时行情，不作为本研究的历史数据源。

冻结参数后复现2025最终检验（需要本机RQData授权数据或已有私有缓存）：

```bash
.venv/bin/python factor/run_alpha_ml_holdout_2025.py
```
