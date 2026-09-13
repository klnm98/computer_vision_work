"""在受限环境里推送代码到远程仓库。

背景：本机所在环境对进程做了限制，导致
  1) git 默认的 schannel TLS 后端取不到系统证书（报 SEC_E_NO_CREDENTIALS）；
  2) 凭据助手（GCM）无法弹出登录窗口。
因此这个脚本：
  * 先把系统信任的根证书导出成 PEM（TLS 被拦截时需要），并用 openssl 后端调用 git；
  * 若存在 ``.git-tmp/token.txt``（里面只有一行 GitHub Personal Access Token），
    则通过临时 askpass 脚本把 token 交给 git —— token 不会出现在命令行、
    也不会写进 git 配置或远程 URL。

用法：
    python tools/push_remote.py                 # 推送当前分支到 origin
    python tools/push_remote.py main            # 推送指定分支
    python tools/push_remote.py --dry-run main   # 只验证鉴权，不改远程
"""
from __future__ import annotations

import argparse
import os
import ssl
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(ROOT, ".git-tmp")
CA_PEM = os.path.join(TMP, "win-ca.pem")
TOKEN_FILE = os.path.join(TMP, "token.txt")
ASKPASS = os.path.join(TMP, "askpass.py")


def export_system_ca() -> str:
    """把系统信任的根证书导出为 PEM（供 git 的 OpenSSL 后端使用）。"""
    os.makedirs(TMP, exist_ok=True)
    certs = ssl.create_default_context().get_ca_certs(binary_form=True)
    import base64

    with open(CA_PEM, "wb") as f:
        for der in certs:
            b64 = base64.encodebytes(der).decode("ascii")
            f.write(b"-----BEGIN CERTIFICATE-----\n")
            for i in range(0, len(b64), 64):
                f.write(b64[i:i + 64].encode("ascii"))
            f.write(b"-----END CERTIFICATE-----\n")
    print(f"[推送] 已导出 {len(certs)} 张系统根证书 -> {CA_PEM}")
    return CA_PEM


def make_askpass() -> str:
    """生成一个 askpass 脚本：从 .git-tmp/token.txt 读取 token 交给 git。"""
    with open(ASKPASS, "w", encoding="utf8") as f:
        f.write(
            "import os, sys\n"
            f"p = r'{TOKEN_FILE}'\n"
            "prompt = (sys.argv[1] if len(sys.argv) > 1 else '').lower()\n"
            "try:\n"
            "    tok = open(p, encoding='utf8').read().strip()\n"
            "except OSError:\n"
            "    tok = ''\n"
            "if not tok:\n"
            "    sys.exit(1)\n"
            "# 用户名提示返回占位符，密码提示返回 token\n"
            "print('x-access-token' if 'username' in prompt and 'password' not in prompt else tok)\n"
        )
    return ASKPASS


def main() -> int:
    ap = argparse.ArgumentParser(description="推送代码到远程仓库（含受限环境 TLS 处理）")
    ap.add_argument("branch", nargs="?", default=None, help="要推送的分支，默认当前分支")
    ap.add_argument("--dry-run", action="store_true", help="只验证鉴权，不真正推送")
    ap.add_argument("--remote", default="origin")
    args = ap.parse_args()

    branch = args.branch
    if not branch:
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    if not branch or branch == "HEAD":
        raise SystemExit("无法确定当前分支，请显式指定，例如：python tools/push_remote.py main")

    ca = export_system_ca()
    cmd = ["git", "-c", "http.sslBackend=openssl", "-c", f"http.sslCAInfo={ca}"]
    env = dict(os.environ, GIT_TERMINAL_PROMPT="1")
    if os.path.exists(TOKEN_FILE) and os.path.getsize(TOKEN_FILE) > 0:
        print(f"[推送] 使用 {TOKEN_FILE} 中的 token 鉴权（不会打印 token 本身）")
        cmd += ["-c", f"core.askPass={make_askpass()}"]
        env["GIT_ASKPASS"] = ASKPASS
        env.pop("GCM_INTERACTIVE", None)
    else:
        print("[推送] 未找到 .git-tmp/token.txt，将使用系统凭据助手（可能弹登录窗口）")
        env["GCM_INTERACTIVE"] = "auto"

    cmd += ["push"]
    if args.dry_run:
        cmd += ["--dry-run"]
    cmd += [args.remote, f"refs/heads/{branch}:refs/heads/{branch}"]

    print("[推送] 执行：", " ".join(c if not c.startswith("http") else c for c in cmd))
    rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
    if rc == 0:
        print(f"[推送] 成功：{args.remote}/{branch}"
              + ("（dry-run，未真正修改远程）" if args.dry_run else ""))
    else:
        print(f"[推送] 失败（退出码 {rc}）。若提示鉴权失败，请把 GitHub Personal Access Token"
              f"（需要 repo 权限）保存到 {TOKEN_FILE} 后重试。")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
