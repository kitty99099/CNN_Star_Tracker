#!/usr/bin/env python3
"""Force-stop other Python processes on this Linux machine.

Run on Ultra96: python3 stop_all_python.py
Preview only:  python3 stop_all_python.py --dry-run
Includes other users' processes when run as root. Unsaved results are lost.
Does not reset the DPU hardware or stop services from restarting processes.
"""

import argparse
import os
from pathlib import Path
import re
import signal
import sys


PYTHON_NAME = re.compile(r"(?:python|pypy)(?:\d+(?:\.\d+)*)?(?:[dm])?$")


def read_process(pid):
    """Return identity and display information, or None for a non-Python PID."""
    directory = Path('/proc') / str(pid)
    # Prefer the executable: argv can contain unrelated Python filenames.
    try:
        name = Path(os.readlink(str(directory / 'exe'))).name
        if name.endswith(' (deleted)'):
            name = name[:-10]
    except PermissionError:
        name = (directory / 'comm').read_text().strip()
    if not PYTHON_NAME.fullmatch(name):
        return None
    stat = (directory / 'stat').read_text()
    fields = stat[stat.rfind(')') + 2:].split()
    if fields[0] == 'Z':
        return None
    started = fields[19]  # /proc/PID/stat field 22: start time.
    try:
        command = (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
    except PermissionError:
        command = name
    return started, command


def kill_process(pid, started):
    """Recheck the process identity before sending SIGKILL."""
    if pid == os.getpid():
        return False
    current = read_process(pid)
    if current is None or current[0] != started:
        return False
    os.kill(pid, signal.SIGKILL)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='List targets without killing anything')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Run this script in the Ultra96 Linux terminal, not on Windows.')
    print('Host: {} | helper PID: {}'.format(os.uname().nodename, os.getpid()), flush=True)
    targets = []
    denied = 0
    for directory in sorted(Path('/proc').iterdir(), key=lambda p: p.name):
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        pid = int(directory.name)
        try:
            process = read_process(pid)
            if process is not None:
                targets.append((pid, process[0], process[1]))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            denied += 1
    if not targets:
        print('No other visible Python processes found.')
    sent = 0
    failed = 0
    for pid, started, command in targets:
        print('{} PID {}: {}'.format('PREVIEW' if args.dry_run else 'TARGET', pid, command), flush=True)
        if args.dry_run:
            continue
        try:
            if kill_process(pid, started):
                sent += 1
                print('  SIGKILL sent.', flush=True)
            else:
                print('  Skipped: process changed or exited.', flush=True)
        except (FileNotFoundError, ProcessLookupError):
            print('  Already exited.', flush=True)
        except PermissionError:
            failed += 1
            print('  Permission denied; run as root to stop other users\' processes.', flush=True)
    if denied:
        print('Could not inspect {} processes due to permissions.'.format(denied))
    if not args.dry_run:
        print('SIGKILL sent to {} processes; {} permission failures.'.format(sent, failed))
        print('Kernel-blocked tasks may remain until the kernel operation returns.')
    return 1 if failed or denied else 0


if __name__ == '__main__':
    sys.exit(main())
