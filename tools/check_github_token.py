"""校验 .git-tmp/token.txt 里的 GitHub token 是否有效、有哪些权限（不打印 token）。"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN_FILE = os.path.join(ROOT, ".git-tmp", "token.txt")
UA = {"User-Agent": "woodblock-uploader", "Accept": "application/vnd.github+json"}


def main() -> int:
    if not os.path.exists(TOKEN_FILE):
        print("找不到 .git-tmp/token.txt")
        return 1
    token = open(TOKEN_FILE, encoding="utf8").read().strip()
    req = urllib.request.Request("https://api.github.com/user",
                                 headers={**UA, "Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            data = json.load(r)
            scopes = r.headers.get("x-oauth-scopes", "")
            print(f"[token] 有效：登录名 {data.get('login')}，类型 {data.get('type')}")
            print(f"[token] 权限范围: {scopes or '(未返回)'}")
            need = {"repo", "public_repo"}
            ok = any(s.strip() in need for s in scopes.split(",")) if scopes else None
            print(f"[token] 是否可写仓库: {'是' if ok else '未知/否'}")
            return 0 if ok is not False else 2
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf8", "ignore")[:200]
        print(f"[token] 校验失败 HTTP {e.code}: {body}")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"[token] 校验异常 {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
