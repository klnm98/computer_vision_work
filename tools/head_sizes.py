"""Print Content-Length for the given URLs."""
import sys
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
for url in sys.argv[1:]:
    try:
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=40) as r:
            n = int(r.headers.get("Content-Length", -1))
        print(f"{n/1e6:10.1f} MB  {url}")
    except Exception as e:  # noqa: BLE001
        print(f"       ERR  {url}  {type(e).__name__} {e}")
