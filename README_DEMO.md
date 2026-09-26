# VeighNa / vnpy_ctp 最小交易系统 Demo

《当代量化交易系统的原理与实现》第二讲作业的 CTP Demo 部分。

本仓库基于 `vnpy 4.4.0` 和 `vnpy_ctp 6.7.7.2`，实现以下最小链路：

1. 连接 SimNow CTP 交易与行情前置；
2. 登录仿真账户；
3. 查询并筛选期货合约；
4. 订阅指定合约行情；
5. 持续打印并保存 Tick 推送；
6. 用前一窗口均价演示上穿/下穿策略信号；
7. 通过 VeighNa UI 查看行情、委托、成交，并支持手工仿真下单。

演示信号**只打印、不自动下单**。价格触发自动下单尚未实现，先按课程要求提交
[设计与风控方案](docs/order-plan.md)，待确认后再开发。

## 当前验证状态

| 环节 | 状态 |
| --- | --- |
| Python/vn.py/vnpy_ctp 导入 | 已验证 |
| SimNow 第二套 7×24 交易柜台登录 | 已验证 |
| 合约查询和行情订阅代码 | 已完成，等待活跃时段实测 |
| Tick 保存和演示信号 | 已完成并通过单元测试 |
| 自动发单 | 未实现 |
| 实盘交易 | 不支持，本项目只用于课堂仿真 |

这里不把尚未实测的功能写成成功结果。运行状态以本机终端日志为准。

## 安全边界

- 仓库不包含任何 UserID、密码、API 密钥或本机 `.vntrader` 配置；
- 登录信息仅在本机终端临时输入，密码输入不可见；
- 行情脚本未导入委托请求类型，不具备下单入口；
- 仅使用获得授权的 SimNow 仿真账号，不得连接真实账户；
- `.vntrader/`、Tick 文件、虚拟环境和日志均被 Git 忽略。

## 安装

建议使用 Python 3.12 和独立虚拟环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

本项目已在 Apple 芯片 Mac 上验证。Mac 如遇 `vnpy_ctp` 编译问题，请参考
[vn.py 官方 Mac 安装指南](https://www.vnpy.com/docs/cn/community/install/mac_install.html)
和 [`vnpy_ctp` 官方仓库](https://github.com/vnpy/vnpy_ctp)。

## 运行

### 1. 只验证柜台登录

```bash
.venv/bin/python ctpdemo/secure_login.py
```

Mac 也可以双击 `connect_simnow.command`。程序仅验证登录，不订阅行情、不发送委托。

### 2. 订阅并打印行情

```bash
.venv/bin/python ctpdemo/market_demo.py --ticks 10 --timeout 90
```

Mac 也可以双击 `subscribe_simnow.command`。登录后输入品种前缀或合约代码，从查询到的
期货合约中选择一个标的。Tick 会打印到终端，并保存至本地 `.vntrader/ticks/`。

### 3. 启动 VeighNa 图形界面

```bash
.venv/bin/python ctpdemo/run.py
```

在 UI 中通过“系统 → 连接 CTP”连接仿真柜台。UI 自带手工委托组件；策略信号仍只在
终端打印，不会自动发送订单。

## 测试

```bash
python3 -m unittest discover -s ctpdemo/tests -p 'test_*.py' -v
```

测试覆盖合约筛选、Tick CSV 字段、逐合约信号隔离、窗口校验和不使用未来 Tick 的
穿越判定。

## 结构

```text
ctpdemo/
├── run.py             # VeighNa UI、行情/日志/委托/成交事件打印
├── secure_login.py    # 隐藏输入凭据，只验证登录
├── market_demo.py     # 合约查询、行情订阅、Tick 打印与保存
├── signals.py         # 不下单的教学信号
└── tests/             # 单元测试
docs/
├── architecture.md    # 对应课堂三层交易系统架构
└── order-plan.md      # 价格触发下单的待确认方案
```

因子挖掘属于作业的另一部分，已经单独完成；方法和结果见
[`factor/results/alpha_ml/alpha_ml_report.md`](factor/results/alpha_ml/alpha_ml_report.md)，不与CTP Demo的实时链路混写。
