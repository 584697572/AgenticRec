"""Run an explicit command and preserve its real output/exit code for A1 evidence."""
import argparse
import datetime
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "reproduction/logs/a1_20261001"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.name):
        parser.error("Invalid log name")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Command required")
    LOG.mkdir(parents=True, exist_ok=True)
    record = {"command": command, "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with (LOG / (args.name + ".stdout.log")).open("wb") as out, (LOG / (args.name + ".stderr.log")).open("wb") as err:
        try:
            result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err)
            code = result.returncode
        except OSError as error:
            err.write((type(error).__name__ + ": " + str(error) + "\n").encode())
            code = 127
    record.update(exit_code=code, finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (LOG / (args.name + ".exitcode.txt")).write_text(str(code) + "\n")
    (LOG / (args.name + ".command.json")).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
