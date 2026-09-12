"""List remote ZIP entry names matching a regex."""
import re
import sys

sys.path.insert(0, r"D:\computer_vision_work\tools")
from remote_zip_entry import central_directory  # noqa: E402

url, pattern = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 15
entries = central_directory(url)
print(f"total entries={len(entries)}")
rx = re.compile(pattern)
hits = [e for e in entries if rx.search(e["name"])]
print(f"matches={len(hits)}")
for e in hits[:limit]:
    print(f"  {e['name']:40s} comp={e['comp']:9d} usize={e['usize']:9d} method={e['method']}")
print("  ... tail sample:")
for e in entries[-limit:]:
    print(f"  {e['name']:40s} comp={e['comp']:9d}")
