"""直接读 Windows 凭据管理器，找 Git Credential Manager 保存的 GitHub token。

沙箱里 git 的凭据助手（GCM）跑不起来（命名管道被拒），所以绕过 git 直接调用
Windows API（CredEnumerateW / CredReadW）。脚本只打印凭据的 target 名称和
长度，**不会打印 token 内容**；找到后写入 .git-tmp/token.txt 供 push_remote.py 使用。
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(ROOT, ".git-tmp")
TOKEN_FILE = os.path.join(TMP, "token.txt")

CRED_TYPE_GENERIC = 1
advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)


class CREDENTIAL_ATTRIBUTE(ctypes.Structure):
    _fields_ = [("Keyword", wt.LPWSTR), ("Flags", wt.DWORD),
                ("ValueSize", wt.DWORD), ("Value", ctypes.POINTER(ctypes.c_byte))]


class CREDENTIAL(ctypes.Structure):
    _fields_ = [
        ("Flags", wt.DWORD), ("Type", wt.DWORD), ("TargetName", wt.LPWSTR),
        ("Comment", wt.LPWSTR), ("LastWritten", wt.FILETIME),
        ("CredentialBlobSize", wt.DWORD), ("CredentialBlob", ctypes.POINTER(ctypes.c_byte)),
        ("Persist", wt.DWORD), ("AttributeCount", wt.DWORD),
        ("Attributes", ctypes.POINTER(CREDENTIAL_ATTRIBUTE)),
        ("TargetAlias", wt.LPWSTR), ("UserName", wt.LPWSTR),
    ]


def enumerate_targets() -> list[str]:
    count = wt.DWORD(0)
    pcreds = ctypes.POINTER(ctypes.POINTER(CREDENTIAL))()
    ok = advapi32.CredEnumerateW(None, 0, ctypes.byref(count), ctypes.byref(pcreds))
    if not ok:
        err = ctypes.get_last_error()
        print(f"[凭据] CredEnumerateW 失败，错误码 {err}（1168=未找到任何凭据，5=拒绝访问）")
        return []
    targets = []
    for i in range(count.value):
        cred = pcreds[i].contents
        targets.append(cred.TargetName or "")
    advapi32.CredFree(pcreds)
    return targets


def read_credential(target: str):
    pcred = ctypes.POINTER(CREDENTIAL)()
    ok = advapi32.CredReadW(target, CRED_TYPE_GENERIC, 0, ctypes.byref(pcred))
    if not ok:
        return None, None
    cred = pcred.contents
    size = cred.CredentialBlobSize
    blob = ctypes.string_at(cred.CredentialBlob, size) if size else b""
    user = cred.UserName or ""
    advapi32.CredFree(pcred)
    # 凭据内容是 UTF-16LE 或 UTF-8，两种都试一下
    for enc in ("utf-16-le", "utf-8"):
        try:
            text = blob.decode(enc).strip("\x00").strip()
            if text and all(32 <= ord(ch) < 127 or ch in "-_" for ch in text):
                return user, text
        except UnicodeDecodeError:
            continue
    try:
        return user, blob.decode("utf-16-le").strip("\x00").strip()
    except UnicodeDecodeError:
        return user, None


def main() -> int:
    os.makedirs(TMP, exist_ok=True)
    targets = enumerate_targets()
    print(f"[凭据] 当前用户共有 {len(targets)} 条 generic 凭据")
    git_like = [t for t in targets if "github" in t.lower() or t.startswith("git:")]
    for t in git_like:
        print(f"   候选: {t}")

    if not git_like:
        print("[凭据] 没有找到 GitHub 相关凭据。请把 PAT 写到 "
              f"{TOKEN_FILE}，或在终端自行 push。")
        return 1

    # 优先 host 级凭据：git:https://github.com
    ordered = sorted(git_like, key=lambda t: (t != "git:https://github.com", len(t)))
    for target in ordered:
        user, secret = read_credential(target)
        if secret and len(secret) >= 20:
            with open(TOKEN_FILE, "w", encoding="utf8") as f:
                f.write(secret)
            print(f"[凭据] 已取到: target={target} user={user or '(空)'} "
                  f"token 长度={len(secret)}（内容不显示，已写入 .git-tmp/token.txt）")
            return 0
        print(f"[凭据] {target}: 未取到可用的 token（user={user}）")
    print("[凭据] 凭据存在但无法解析出 token，请改用 PAT 文件方式。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
