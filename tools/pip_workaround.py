"""Run pip with a sandbox workaround.

This sandbox denies file creation inside directories created with POSIX mode 0o700
(what ``tempfile.mkdtemp`` requests), which breaks pip's unpack/target temp dirs.
Patching ``os.mkdir`` to request 0o777 makes those directories writable; pip is then
invoked normally with the remaining CLI arguments.
"""
import os
import sys

_real_mkdir = os.mkdir


def _mkdir(path, mode=0o777, *, dir_fd=None):
    return _real_mkdir(path, 0o777, dir_fd=dir_fd)


os.mkdir = _mkdir

from pip._internal.cli.main import main  # noqa: E402

sys.exit(main(sys.argv[1:]))
