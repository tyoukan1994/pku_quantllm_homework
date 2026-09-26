"""Submit a Choice manual-activation request without storing the password."""

from getpass import getpass

from EmQuantAPI import c


def main() -> int:
    print("Choice QuantAPI 人工激活申请")
    print("账号和密码仅传给 Choice 官方 SDK；密码不会显示或写入本项目。")
    username = input("Choice 账号/绑定手机号：").strip()
    password = getpass("Choice 密码（隐藏输入）：")
    email = input("接收 userInfo 令牌的有效邮箱：").strip()

    result = c.manualactivate(username, password, f"email={email}")
    password = ""

    print(f"激活结果代码：{result.ErrorCode}")
    if result.ErrorCode:
        print(f"激活失败：{result.ErrorMsg}")
        return 1

    print("人工激活申请已提交。请等待 Choice 客户经理向该邮箱发送 userInfo 文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
