"""Terminate the Lambda instance this script runs on (via Cloud API).

Reads LAMBDA_API_KEY from ./.env; identifies its own instance by public
IP. Used by run_all_and_terminate.sh's exit trap.

Usage: python3 scripts/lambda/self_terminate.py [--dry-run]
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

API = "https://cloud.lambdalabs.com/api/v1"


def _env(key: str) -> str:
    for line in open(".env"):
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"{key} not in .env")


def _req(path: str, token: str, body: dict | None = None):
    req = urllib.request.Request(
        API + path,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json",
                 "User-Agent": "mast-p2-selfterminate"},
        data=json.dumps(body).encode() if body is not None else None,
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main() -> None:
    token = _env("LAMBDA_API_KEY")
    my_ip = urllib.request.urlopen(
        "https://checkip.amazonaws.com", timeout=15).read().decode().strip()
    instances = _req("/instances", token)["data"]
    mine = [i for i in instances if i.get("ip") == my_ip]
    if not mine:
        raise SystemExit(f"no instance with ip {my_ip} found ({len(instances)} listed)")
    iid = mine[0]["id"]
    print(f"instance {iid} ({my_ip})", flush=True)
    if "--dry-run" in sys.argv:
        print("dry run — not terminating")
        return
    out = _req("/instance-operations/terminate", token,
               {"instance_ids": [iid]})
    print("terminate requested:", json.dumps(out)[:200], flush=True)


if __name__ == "__main__":
    main()
