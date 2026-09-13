"""发布 GitHub Release 并把 APK 作为附件上传。

凭据来源（按优先级）：
  1) 环境变量 GITHUB_TOKEN / GH_TOKEN
  2) .git-tmp/token.txt（一行 Personal Access Token）
token 不会被打印，也不会写进 git 配置。

用法:
    python tools/release_apk.py --check                 # 只验证 token 与仓库权限
    python tools/release_apk.py --dry-run               # 打印将要执行的动作
    python tools/release_apk.py                         # 创建/更新 Release 并上传 APK
    python tools/release_apk.py --tag v1.0.0 --name "标题" --notes-file RELEASE_NOTES.md
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN_FILE = os.path.join(ROOT, ".git-tmp", "token.txt")
DIST = os.path.join(ROOT, "dist")
API = "https://api.github.com"
UPLOADS = "https://uploads.github.com"
UA = "wood-block-release-script"


def read_token() -> str | None:
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        return tok.strip()
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, encoding="utf8") as f:
            data = f.read().strip()
        if data:
            return data
    return None


def repo_slug() -> str:
    """从 git remote 解析 owner/repo。"""
    import subprocess

    url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    if url.endswith(".git"):
        url = url[:-4]
    if url.startswith("git@"):                     # git@github.com:owner/repo
        return url.split(":", 1)[1]
    return "/".join(url.split("/")[-2:])


def api(path: str, token: str, method: str = "GET", data: dict | None = None,
        raw: bytes | None = None, content_type: str = "application/json"):
    url = path if path.startswith("http") else API + path
    body = raw if raw is not None else (json.dumps(data).encode() if data is not None else None)
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", UA)
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if body is not None:
        req.add_header("Content-Type", content_type)
    with urllib.request.urlopen(req, timeout=120) as r:
        payload = r.read()
        return r.status, (json.loads(payload) if payload else {})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1.0.0")
    ap.add_argument("--name", default=None, help="Release 标题，默认用 tag")
    ap.add_argument("--notes-file", default=os.path.join(ROOT, "RELEASE_NOTES.md"))
    ap.add_argument("--assets", nargs="*", default=None,
                    help="要上传的文件，默认 dist/ 下所有文件")
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--prerelease", action="store_true")
    ap.add_argument("--check", action="store_true", help="只验证 token 与权限")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = read_token()
    if not token:
        print("找不到凭据：请把 GitHub Personal Access Token 放到 "
              f"{TOKEN_FILE}（一行），或设置环境变量 GITHUB_TOKEN。")
        return 2
    slug = repo_slug()
    print(f"[发布] 仓库: {slug}   tag: {args.tag}")

    # ---- 验证 token
    try:
        _, me = api("/user", token)
        print(f"[发布] token 属于: {me.get('login')}")
    except urllib.error.HTTPError as e:
        print(f"[发布] token 验证失败: HTTP {e.code} {e.reason}")
        return 3
    try:
        _, repo = api(f"/repos/{slug}", token)
        perms = repo.get("permissions", {})
        print(f"[发布] 仓库权限: push={perms.get('push')} admin={perms.get('admin')} "
              f"private={repo.get('private')}")
        if not perms.get("push"):
            print("[发布] 该 token 没有 push 权限，无法创建 Release。")
            return 3
    except urllib.error.HTTPError as e:
        print(f"[发布] 读取仓库失败: HTTP {e.code} —— token 可能没有 repo 权限或仓库不存在")
        return 3

    if args.check:
        print("[发布] 权限检查通过。")
        return 0

    assets = args.assets
    if assets is None:
        assets = [os.path.join(DIST, f) for f in sorted(os.listdir(DIST))] if os.path.isdir(DIST) else []
    assets = [a for a in assets if os.path.exists(a)]
    print(f"[发布] 附件: {[os.path.basename(a) for a in assets]}")

    body = ""
    if os.path.exists(args.notes_file):
        body = open(args.notes_file, encoding="utf8").read()
        print(f"[发布] 发布说明: {args.notes_file} ({len(body)} 字符)")
    else:
        print(f"[发布] 未找到发布说明文件 {args.notes_file}，将使用空说明")

    if args.dry_run:
        print("[发布] dry-run：将创建（或更新）Release 并上传上述附件，未执行。")
        return 0

    # ---- 创建或更新 Release
    payload = {"tag_name": args.tag, "name": args.name or args.tag, "body": body,
               "draft": args.draft, "prerelease": args.prerelease}
    try:
        status, rel = api(f"/repos/{slug}/releases", token, "POST", payload)
        print(f"[发布] 已创建 Release: {rel.get('html_url')}")
        release_id = rel["id"]
        existing = {a["name"] for a in rel.get("assets", [])}
    except urllib.error.HTTPError as e:
        if e.code != 422:      # 422 = 该 tag 已存在 Release
            print(f"[发布] 创建失败: HTTP {e.code} {e.read()[:300]!r}")
            return 4
        status, rel = api(f"/repos/{slug}/releases/tags/{args.tag}", token)
        release_id = rel["id"]
        existing = {a["name"] for a in rel.get("assets", [])}
        api(f"/repos/{slug}/releases/{release_id}", token, "PATCH",
            {"name": payload["name"], "body": body})
        print(f"[发布] Release 已存在，已更新说明: {rel.get('html_url')}")

    # ---- 上传附件（已存在的同名附件会先删除再上传，便于重复执行）
    for path in assets:
        name = os.path.basename(path)
        if name in existing:
            for a in rel.get("assets", []):
                if a["name"] == name:
                    api(f"/repos/{slug}/releases/assets/{a['id']}", token, "DELETE")
                    print(f"[发布] 已移除旧附件 {name}")
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        data = open(path, "rb").read()
        url = (f"{UPLOADS}/repos/{slug}/releases/{release_id}/assets"
               f"?name={urllib.parse.quote(name)}")
        try:
            _, res = api(url, token, "POST", raw=data, content_type=ctype)
            print(f"[发布] 已上传 {name} ({len(data)/1e6:.1f} MB) -> "
                  f"{res.get('browser_download_url')}")
        except urllib.error.HTTPError as e:
            print(f"[发布] 上传 {name} 失败: HTTP {e.code} {e.read()[:200]!r}")
            return 5

    _, rel = api(f"/repos/{slug}/releases/{release_id}", token)
    print(f"\n[发布] 完成：{rel.get('html_url')}")
    print("[发布] 页面上的 Assets 里就能下载 APK。")
    return 0


if __name__ == "__main__":
    import urllib.parse  # noqa: E402  (仅上传附件时用到)

    raise SystemExit(main())
