# 因子研究与可复现入口

本目录已经完成RQData月度基本面复合因子和Alpha158机器学习对照。研究记录继续遵守：

1. 数据来源与许可、品种、合约拼接方法、时间频率、时区、样本起止及缺失值。
2. 因子假设和公式；只使用当时已经可见的数据，避免未来信息泄漏。
3. 因子值分布、方向、稳定性和样本外表现；报告所有尝试过的候选，避免只展示成功者。
4. 交易规则、手续费、滑点、换月处理与资金管理。
5. 周一至周四每天的实际运行记录。如果是仿真前向测试，明确标为“前向测试”，不要写成历史回测。

建议先用一份可合法使用的历史 K 线 CSV 验证数据管线，再决定具体因子。只有四天的前向测试不足以证明因子有效。

## 当前候选与可替换数据入口

`data_source.py` 定义统一的 `BarDataSource` 接口，并已实现 `CsvBarDataSource` 和 `JqDataSource`。当前先使用聚宽 JQData；Choice QuantAPI 权限开通后仍可新增 Choice 实现，无需修改因子和回测代码。标准字段为：`datetime,symbol,open,high,low,close,volume`。

当前候选因子是 20 日波动率调整时间序列动量：过去 20 日对数收益除以同期已实现波动率。信号在第 t 日收盘计算，只用于第 t+1 日收益，并按仓位变化收取交易成本。这只是待数据验证的研究假设，不是已发现的有效因子。

新增的主研究方案是基于 Fama–French 三因子思想的规模—价值复合因子，具体方法见 [`ff3_combo_protocol.md`](ff3_combo_protocol.md)。探索性绩效和调参样本为2015—2024年，允许比较全部35组预先限定的离散权重并保留完整记录；参数冻结后，2025年被一次性解封用于最终检验。`research_policy.py`分别约束开发期与最终检验窗口，拒绝超出授权边界的数据请求。

机器学习开发报告见[`results/alpha_ml/alpha_ml_report.md`](results/alpha_ml/alpha_ml_report.md)，最终报告见[`results/alpha_ml/final_factor_mining_report.md`](results/alpha_ml/final_factor_mining_report.md)。研究加入158项Alpha158价量特征、4项基本面/技术特征和4项课件公式因子，并比较Ridge、LightGBM、MLP及三模型等权秩集成。2020—2024结果是可反复查看的开发期内部证据；2025年1—12月收益仅用于冻结方案的一次性最终评价。最终平均Rank IC为0.0256，但证据不显著且扣费多空收益为负，因此未宣称因子通过验证。

已有本地缓存时，可运行：

```bash
PYTHONPATH=factor .venv/bin/python factor/run_alpha_ml.py
```

如需重新下载日频特征窗口：

```bash
PYTHONPATH=factor .venv/bin/python factor/run_alpha_ml.py \
  --refresh-daily --refresh-features
```

原始行情、License和股票级特征均位于Git忽略目录。脚本会分别读取未复权成交量/成交额和价格复权因子，确保VWAP与前复权OHLC口径一致。

## 米筐 RQData 历史数据

项目虚拟环境已安装 `rqdatac` 和 `vnpy_rqdata`。当前 License 已实测可读取沪深300历史日线，以及 A 股时点化季度财务数据（包括营业收入、净利润和公告日期）。其他品种、频率、字段和时间范围仍取决于账号权限。RQData 不会取代 SimNow/CTP 的仿真柜台和报单功能。

可双击项目根目录的 `check_rqdata.command`。如果米筐安装包仍位于本机下载目录，检查程序会只解析 `make.sh` 内嵌的 License，不执行脚本，也不会把 License 写入 shell 配置；如果找不到安装包，才会提示隐藏输入凭据。程序只查询 2024 年初的一小段沪深300日线，以验证登录和历史行情权限。不要运行或上传外部安装包中含明文授权信息的 `make.sh`、`make.bat`。

也可在终端运行：

```bash
.venv/bin/python factor/check_rqdata.py
```

权限验证成功后，再通过 `rqdatac` 下载研究数据并标准化为 `datetime,symbol,open,high,low,close,volume`。原始行情保存在被 Git 忽略的 `data/private/`，不得随作业代码上传。

## 聚宽历史数据

已在项目虚拟环境安装 `jqdatasdk`。Mac 用户可双击项目根目录的
`download_jqdata.command`，在终端中输入聚宽账号和隐藏密码。脚本默认下载上证50、
沪深300、中证500和中证1000从 2015 年至今的日线，保存到被 Git 忽略的
`data/private/jqdata/broad_indices_daily.csv`。账号和密码不会写入文件。

也可在终端中指定区间或代码：

```bash
.venv/bin/python factor/download_jqdata.py \
  --symbols 000016.XSHG 000300.XSHG 000905.XSHG 000852.XSHG \
  --start 2015-01-01 \
  --end 2026-09-20
```

聚宽数据只用于历史研究；SimNow/CTP 继续用于实时行情订阅和仿真柜台验证。

CSV 运行示例：

```bash
.venv/bin/python factor/run_factor.py \
  --csv data/example.csv \
  --symbol rb.main \
  --lookback 20 \
  --cost-bps 2
```

提交前仍需明确连续合约拼接、换月成本、样本内/样本外切分，以及周一至周四是历史回测更新还是仿真前向测试。
