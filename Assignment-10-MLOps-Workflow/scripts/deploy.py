#!/usr/bin/env python
"""Cross-platform local deployment simulation (the CD part of CI/CD).

Runs the standard deployment sequence against the local Docker engine:

1. build the Docker image (tagged with the central app version)
2. remove any stale container, then run a fresh one
3. poll ``GET /health`` until the service is ready
4. send one validation ``POST /predict``
5. report deployment success / clean up on failure

The only "deployment" performed is to the local Docker engine -- this is
an educational CD *simulation*, not a cloud release.  Docker must be
running; otherwise the script fails honestly with a clear message.

Usage:
    python scripts/deploy.py            # full build + run + verify
    python scripts/deploy.py --verify   # only verify an already-running container
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402

HEALTH_URL = f"http://localhost:{config.API_PORT}/health"
PREDICT_URL = f"http://localhost:{config.API_PORT}/predict"
SAMPLE_VEHICLE = {
    "vehicle_age": 5,
    "km_driven": 45_000,
    "mileage_kmpl": 18.5,
    "engine_cc": 1_498,
    "max_power_bhp": 100,
}
STARTUP_TIMEOUT_S = 90.0
POLL_INTERVAL_S = 2.0


def _run(cmd: list[str]) -> None:
    """Run a command, streaming output, raising on failure."""
    print(f"+ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {cmd}")


def docker_build() -> None:
    _run(["docker", "build", "-t", config.DOCKER_IMAGE, str(PROJECT_ROOT)])


def docker_run() -> None:
    # Remove a stale container from a previous simulation, if present.
    subprocess.run(
        ["docker", "rm", "-f", config.DOCKER_CONTAINER],
        capture_output=True,
        check=False,
    )
    _run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            config.DOCKER_CONTAINER,
            "-p",
            f"{config.API_PORT}:{config.API_PORT}",
            config.DOCKER_IMAGE,
        ]
    )


def wait_until_healthy(timeout_s: float = STARTUP_TIMEOUT_S) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=5) as response:
                if response.status == 200:
                    payload = json.loads(response.read().decode("utf-8"))
                    if payload.get("status") == "healthy":
                        print(f"service healthy: {payload}")
                        return
        except Exception as exc:  # connection refused while starting up
            last_error = exc
        time.sleep(POLL_INTERVAL_S)
    raise RuntimeError(f"service did not become healthy in {timeout_s:.0f}s: {last_error}")


def validate_predict() -> float:
    body = json.dumps(SAMPLE_VEHICLE).encode("utf-8")
    request = urllib.request.Request(
        PREDICT_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    print(f"validation prediction: {payload}")
    value = payload.get("prediction")
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"non-numeric prediction in response: {payload}")
    return float(value)


def teardown() -> None:
    subprocess.run(
        ["docker", "rm", "-f", config.DOCKER_CONTAINER],
        capture_output=True,
        check=False,
    )


def main() -> int:
    print("=== Local deployment simulation (CD) ===")
    verify_only = "--verify" in sys.argv
    try:
        if not verify_only:
            docker_build()
            docker_run()
        wait_until_healthy()
        validate_predict()
    except RuntimeError as exc:
        message = str(exc)
        print(f"DEPLOYMENT FAILED: {message}")
        if "docker" in message.lower() and not verify_only:
            print(
                "Hint: is Docker Desktop installed *and* running? "
                "The daemon must be reachable for the build/run steps."
            )
        teardown()
        return 1
    except urllib.error.URLError as exc:
        print(f"DEPLOYMENT FAILED (network): {exc}")
        teardown()
        return 1
    print(
        f"DEPLOYMENT SUCCESS: {config.DOCKER_IMAGE} serving on "
        f"http://localhost:{config.API_PORT} (local simulation)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
