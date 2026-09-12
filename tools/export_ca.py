"""把 Python/系统信任的根证书导出为 PEM，供 git(OpenSSL 后端) 使用。

在 TLS 被拦截的环境里，git 自带的 mozilla CA 列表不包含拦截根证书，会报
"unable to get local issuer certificate"；而 Python 使用的是系统证书库，
因此把系统库导出即可让 git 正常校验。
"""
from __future__ import annotations

import os
import ssl

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   ".git-tmp", "win-ca.pem")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

ctx = ssl.create_default_context()
certs = ctx.get_ca_certs(binary_form=True)
with open(OUT, "wb") as f:
    for der in certs:
        b64 = __import__("base64").encodebytes(der).decode("ascii")
        f.write(b"-----BEGIN CERTIFICATE-----\n")
        for i in range(0, len(b64), 64):
            f.write(b64[i:i + 64].encode("ascii"))
        f.write(b"-----END CERTIFICATE-----\n")
print(f"已导出 {len(certs)} 张证书 -> {OUT} ({os.path.getsize(OUT)} 字节)")
