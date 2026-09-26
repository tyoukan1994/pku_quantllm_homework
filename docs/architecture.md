# CTP Demo 架构映射

本 Demo 对应课堂中的“底层接口—中层引擎—策略应用”三层结构。

| 层级 | 课堂概念 | 本 Demo 对应实现 |
| --- | --- | --- |
| 底层接口 | 数据结构标准化 | `TickData`、`ContractData`、`SubscribeRequest` |
| 底层接口 | 业务流程标准化 | `CtpGateway.connect`、`subscribe`、回调事件 |
| 中层引擎 | 事件总线 | `EventEngine` 分发日志、合约、行情、委托和成交事件 |
| 中层引擎 | OMS 缓存与指令路由 | VeighNa `MainEngine` 和内置 UI；本 Demo 不重写 OMS |
| 策略应用 | 单标的时序信号 | `TickSignal` 对前一窗口均价做穿越判断 |
| 交易员 UI | 监控与手工委托 | VeighNa `MainWindow` |

## 只读行情链路

```text
SimNow 行情前置
        ↓
    CtpGateway
        ↓
    EventEngine
        ↓
EVENT_CONTRACT / EVENT_TICK
        ↓
合约选择 → Tick 打印/CSV → TickSignal
```

## 已实现与未实现

- 已实现：柜台连接、登录、合约查询、订阅请求、行情回调、Tick 保存、演示信号；
- 复用框架：OMS 缓存、图形界面、手工委托、委托和成交事件；
- 未实现：生产级风控、自动委托路由、数据库、旁路监控、多策略组合管理；
- 禁止：将课堂 Demo 连接真实账户，或把未实测功能写成已验证结果。
