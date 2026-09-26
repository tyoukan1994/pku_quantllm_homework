"""Activate Choice QuantAPI using the official SMS uplink flow."""

from EmQuantAPI import c


def main() -> int:
    print("请先用绑定手机号发送短信 SXDL 到 9535711。")
    print("短信发送后 10 分钟内完成本步骤；运营商可能收取普通短信费。")
    phone = input("绑定手机号：").strip()
    if not (phone.isdigit() and len(phone) == 11):
        print("手机号格式不正确。")
        return 2

    result = c.start(f"LoginMode=SXDL,PhoneNumber={phone},ForceLogin=1")
    print(f"短信激活结果代码：{result.ErrorCode}")
    if result.ErrorCode:
        print(f"短信激活失败：{result.ErrorMsg}")
        return 1

    print("短信激活成功，Choice 已生成本机 userInfo 令牌。")
    c.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
