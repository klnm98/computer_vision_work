"""探测是否已有可用的 GitHub 凭据（优先取上次手动 push 时保存的）。

若取到，会把 token 写到 .git-tmp/token.txt 供 tools/push_remote.py 使用；
脚本只打印用户名和长度，**不会输出 token 本身**。
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(ROOT, ".git-tmp")
TOKEN_FILE = os.path.join(TMP, "token.txt")


def main() -> int:
    os.makedirs(TMP, exist_ok=True)
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "never"

    proc = subprocess.run(["git", "credential", "fill"], cwd=ROOT, env=env,
                          input="protocol=https\nhost=github.com\n\n",
                          capture_output=True, text=True, timeout=60)
    out = proc.stdout or ""
    err = (proc.stderr or "").strip()
    fields = dict(
        line.split("=", 1) for line in out.splitlines() if "=" in line
    )
    user = fields.get("username", "")
    pwd = fields.get("password", "")

    if pwd:
        with open(TOKEN_FILE, "w", encoding="utf8") as f:
            f.write(pwd)
        print(f"[凭据] 取到 GitHub 凭据：username={user or '(空)'}，token 长度 {len(pwd)}"
              f"（已写入 .git-tmp/token.txt，不会显示内容）")
        return 0
    print("[凭据] 没有取到可用凭据。")
    if err:
        print("[凭据] git 提示: " + err[:300])
    print("[凭据] 需要你二选一：\n"
          "  A) 创建一个有 repo 权限的 Personal Access Token，保存到 "
          f"{TOKEN_FILE}（只放 token 一行）；\n"
          "  B) 自己在终端执行 git push / gh release create。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
