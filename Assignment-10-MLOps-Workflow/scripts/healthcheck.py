"""Deployment health check.

Calls ``GET /health`` on the running API, verifies HTTP 200 and that the
JSON body reports ``status == "healthy"``.  Exits 0 on success and 1 on
any failure -- suitable for Docker HEALTHCHECK, CI and the deploy script.

Usage:
    python scripts/healthcheck.py [base_url]

Defaults to http://localhost:8000.  The base URL can also be given via
the API_BASE_URL environment variable.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

DEFAULT_BASE_URL = "http://localhost:8000"


def check_health(base_url: str, timeout: float = 5.0) -> dict:
    """Return the parsed /health JSON or raise on any problem."""
    url = f"{base_url.rstrip('/')}/health"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected HTTP status {response.status}")
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") != "healthy":
        raise RuntimeError(f"service not healthy: {payload}")
    return payload


def main() -> int:
    base_url = os.environ.get("API_BASE_URL", DEFAULT_BASE_URL)
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    try:
        payload = check_health(base_url)
    except Exception as exc:
        print(f"HEALTH CHECK FAILED ({base_url}/health): {exc}")
        return 1
    print(f"HEALTH CHECK OK ({base_url}/health): {payload}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
