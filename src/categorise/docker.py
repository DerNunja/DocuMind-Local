from __future__ import annotations

import shutil
import subprocess


POSTGRES_CONTAINER = "documind-postgres"
POSTGRES_IMAGE = "pgvector/pgvector:pg16"


def ensure_postgres_docker() -> dict[str, str | bool]:
    if shutil.which("docker") is None:
        return {"ok": False, "message": "Docker CLI wurde nicht gefunden."}

    try:
        inspect = subprocess.run(
            ["docker", "inspect", POSTGRES_CONTAINER],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "message": f"Docker konnte nicht geprüft werden: {exc}"}

    try:
        if inspect.returncode == 0:
            start = subprocess.run(
                ["docker", "start", POSTGRES_CONTAINER],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            if start.returncode == 0:
                return {"ok": True, "message": f"Container {POSTGRES_CONTAINER} läuft."}
            message = (start.stderr or start.stdout).strip()
            return {"ok": False, "message": f"Container konnte nicht gestartet werden: {message}"}

        run = subprocess.run(
            [
                "docker",
                "run",
                "--name",
                POSTGRES_CONTAINER,
                "-e",
                "POSTGRES_PASSWORD=postgres",
                "-e",
                "POSTGRES_DB=documind",
                "-p",
                "5432:5432",
                "-d",
                POSTGRES_IMAGE,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if run.returncode == 0:
            return {"ok": True, "message": f"Container {POSTGRES_CONTAINER} wurde angelegt und gestartet."}
        message = (run.stderr or run.stdout).strip()
        return {"ok": False, "message": f"Container konnte nicht angelegt werden: {message}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "message": "Docker-Aufruf hat zu lange gedauert."}
