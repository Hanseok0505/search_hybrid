from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import httpx


DEFAULT_SERVICES = [
    "elasticsearch",
    "redis",
    "neo4j",
    "milvus-standalone",
    "ollama",
    "tika",
]

PORT_CHECKS: Dict[str, int] = {
    "elasticsearch": 9200,
    "redis": 6379,
    "neo4j": 7687,
    "milvus": 19530,
    "ollama": 11434,
    "tika": 9998,
}


@dataclass
class ProbeResult:
    ok: bool
    message: str


def _ok(msg: str) -> ProbeResult:
    return ProbeResult(ok=True, message=msg)


def _fail(msg: str) -> ProbeResult:
    return ProbeResult(ok=False, message=msg)


def _check_file(path: Path) -> ProbeResult:
    if path.exists():
        return _ok(f".env exists: {path}")
    return _fail(f".env not found: {path}. cp .env.example .env before startup.")


def _run_compose_list(project_root: Path) -> Optional[List[str]]:
    try:
        proc = subprocess.run(
            ["docker", "compose", "ps", "--services", "--filter", "status=running"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return []
    services = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return services


def _check_compose_services(project_root: Path, strict: bool) -> List[ProbeResult]:
    running = _run_compose_list(project_root)
    if running is None:
        return [
            _fail("docker executable not found. Skip compose checks or run checks on a machine with Docker.")
        ]
    if running == []:
        return [
            _fail(
                "No running compose services found. Run `docker compose up -d` and wait until dependent services are healthy."
            )
        ]

    out: List[ProbeResult] = []
    for svc in DEFAULT_SERVICES:
        if svc in running:
            out.append(_ok(f"{svc} running"))
        elif strict:
            out.append(_fail(f"{svc} not running"))
        else:
            out.append(_fail(f"{svc} not running (non-strict allowed)"))
    return out


def _socket_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _check_ports() -> List[ProbeResult]:
    out: List[ProbeResult] = []
    for name, port in PORT_CHECKS.items():
        if _socket_reachable("127.0.0.1", port):
            out.append(_ok(f"{name}:{port} reachable"))
        else:
            out.append(_fail(f"{name}:{port} not reachable (host networking check)"))
    return out


async def _check_api(api_base: str) -> List[ProbeResult]:
    out: List[ProbeResult] = []
    async with httpx.AsyncClient(timeout=5.0) as c:
        # health
        try:
            health = await c.get(urljoin(api_base.rstrip("/") + "/", "v1/health"))
            if health.status_code == 200:
                payload = health.json()
                overall = payload.get("status", "unknown")
                if overall in {"ok", "degraded"}:
                    out.append(_ok(f"{api_base}/v1/health {overall}: {payload.get('elastic')}, {payload.get('milvus')}, {payload.get('graph')}, {payload.get('redis')}"))
                else:
                    out.append(_fail(f"{api_base}/v1/health down: {json.dumps(payload, ensure_ascii=False)}"))
            else:
                out.append(_fail(f"{api_base}/v1/health -> HTTP {health.status_code}"))
        except Exception as exc:
            out.append(_fail(f"{api_base}/v1/health failed: {exc}"))

        # source+model endpoints
        for path in ("/v1/sources", "/v1/models?provider=ollama"):
            try:
                r = await c.get(urljoin(api_base.rstrip("/") + "/", path.lstrip("/")))
                if r.status_code == 200:
                    out.append(_ok(f"{path} reachable"))
                else:
                    out.append(_fail(f"{path} failed HTTP {r.status_code}"))
            except Exception as exc:
                out.append(_fail(f"{path} failed: {exc}"))
    return out


def _print_header(lines: List[ProbeResult], label: str, level: str) -> None:
    print(f"\n[{level}] {label}")
    for item in lines:
        prefix = "OK" if item.ok else "WARN"
        print(f"- {prefix}: {item.message}")


def _print_summary(results: List[ProbeResult], strict: bool) -> int:
    ok_count = sum(1 for r in results if r.ok)
    fail_count = len(results) - ok_count
    print(f"\nSummary: {ok_count} passed, {fail_count} warnings/errors")
    if strict and fail_count:
        print("strict mode: failed due to one or more required items\n")
        return 1
    return 0


async def run(project_root: Path, api_base: str, strict: bool, skip_compose: bool, skip_ports: bool) -> int:
    all_results: List[ProbeResult] = []
    all_results.append(_check_file(project_root / ".env"))

    if not skip_compose:
        all_results.extend(_check_compose_services(project_root, strict))

    if not skip_ports:
        all_results.extend(_check_ports())

    api_results = await _check_api(api_base)
    all_results.extend(api_results)

    _print_header([all_results[i] for i in range(min(1, len(all_results)))], ".env", "pre")
    if not skip_compose:
        _print_header(all_results[1:1 + len(DEFAULT_SERVICES)] if len(all_results) >= 1 + len(DEFAULT_SERVICES) else [], "compose", "core")
    start_idx = 1 + (0 if skip_compose else len(DEFAULT_SERVICES))
    if not skip_ports:
        _print_header(all_results[start_idx:start_idx + len(PORT_CHECKS)], "ports", "host")
        start_idx += len(PORT_CHECKS)
    _print_header(all_results[start_idx:], "api", "app")

    return _print_summary(all_results, strict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight runtime check for hybrid search stack")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--api-base", default="http://localhost:8080", help="Base URL of API endpoint")
    parser.add_argument("--strict", action="store_true", help="Fail when any required item fails")
    parser.add_argument("--skip-compose", action="store_true", help="Skip docker-compose service checks")
    parser.add_argument("--skip-ports", action="store_true", help="Skip raw host port checks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exit_code = asyncio.run(
        run(
            project_root=Path(args.project_root).resolve(),
            api_base=args.api_base,
            strict=args.strict,
            skip_compose=args.skip_compose,
            skip_ports=args.skip_ports,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    import asyncio

    main()
