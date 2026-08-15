"""Local web frontend server for ChainCheck.

This intentionally uses only the Python standard library so the UI works with
the existing project dependencies. It serves static files, exposes processed
artifacts as JSON, and runs pipeline.py as a background job.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

HOST = "0.0.0.0"
PORT = 8765
DEFAULT_PROVIDER = "ollama"
DEFAULT_OLLAMA_MODEL = "phi3:latest"
PYTHON_EXECUTABLE = (
    PROJECT_ROOT / ".venv" / "bin" / "python"
    if (PROJECT_ROOT / ".venv" / "bin" / "python").exists()
    else Path(sys.executable)
)

ARTIFACTS = {
    "questions": "questions.json",
    "template_questions": "due_diligence_questions.json",
    "risks": "vc_risk_report.json",
    "audit": "audited_vc_report.json",
    "evidence": "structured_evidence.json",
    "contradictions": "contradiction_evidence.json",
    "graph": "fused_knowledge_graph.json",
    "eval": "eval_results.json",
}


class PipelineJob:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.returncode: int | None = None
        self.command: list[str] = []
        self.logs: list[str] = []

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "returncode": self.returncode,
                "command": self.command,
                "logs": self.logs[-600:],
            }

    def start(self, command: list[str]) -> bool:
        with self.lock:
            if self.running:
                return False
            self.running = True
            self.started_at = time.time()
            self.finished_at = None
            self.returncode = None
            self.command = command
            self.logs = ["$ " + " ".join(command)]

        thread = threading.Thread(target=self._run, args=(command,), daemon=True)
        thread.start()
        return True

    def _run(self, command: list[str]) -> None:
        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                with self.lock:
                    self.logs.append(line.rstrip())
            returncode = process.wait()
        except Exception as exc:  # pragma: no cover - surfaced in UI
            returncode = -1
            with self.lock:
                self.logs.append(f"Server failed to launch pipeline: {exc}")

        with self.lock:
            self.running = False
            self.finished_at = time.time()
            self.returncode = returncode
            self.logs.append(f"Pipeline exited with code {returncode}")


JOB = PipelineJob()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _file_info(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    return {
        "exists": True,
        "size": path.stat().st_size,
        "modified_at": path.stat().st_mtime,
    }


def _processed_summary() -> dict:
    summary: dict[str, object] = {}

    parsed = sorted(PROCESSED_DIR.glob("*_parsed.json"))
    if parsed:
        data = _read_json(parsed[0])
        summary["pitch"] = {
            "file": parsed[0].name,
            "source": data.get("source_file"),
            "statistics": data.get("statistics", {}),
        }

    graph = _read_json(PROCESSED_DIR / "fused_knowledge_graph.json")
    if graph:
        edges = graph.get("links", graph.get("edges", []))
        summary["graph"] = {
            "nodes": len(graph.get("nodes", [])),
            "edges": len(edges),
            "metadata": graph.get("metadata", {}),
        }

    risks = _read_json(PROCESSED_DIR / "vc_risk_report.json")
    if risks:
        summary["risks"] = risks.get("metadata", {})

    questions = _read_json(PROCESSED_DIR / "questions.json")
    if questions:
        summary["questions"] = questions.get("metadata", {})

    audit = _read_json(PROCESSED_DIR / "audited_vc_report.json")
    if audit:
        summary["audit"] = {"total_items": len(audit.get("audited_items", []))}

    return summary


def _default_inputs() -> dict:
    pitch_files = sorted((DATA_DIR / "raw" / "pitch_decks").glob("*.pdf"))
    repo_dirs = [
        p for p in sorted((DATA_DIR / "raw" / "repositories").iterdir())
        if p.is_dir()
    ] if (DATA_DIR / "raw" / "repositories").exists() else []
    return {
        "pitch": str(pitch_files[0]) if pitch_files else "",
        "repo": str(repo_dirs[0]) if repo_dirs else str(PROJECT_ROOT),
        "patents": str(DATA_DIR / "raw" / "patents"),
        "ground_truth": str(PROJECT_ROOT / "eval" / "ground_truth.json"),
    }


def _runtime_info() -> dict:
    checks = {
        "spacy": "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('spacy') else 1)",
        "sentence_transformers": (
            "import importlib.util; "
            "raise SystemExit(0 if importlib.util.find_spec('sentence_transformers') else 1)"
        ),
    }
    packages = {}
    for name, script in checks.items():
        result = subprocess.run(
            [str(PYTHON_EXECUTABLE), "-c", script],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        packages[name] = result.returncode == 0

    ollama_models = _ollama_models(timeout=0.4)

    return {
        "python": str(PYTHON_EXECUTABLE),
        "packages": packages,
        "ollama_reachable": bool(ollama_models),
        "ollama_models": ollama_models,
    }


def _ollama_models(timeout: float = 1.0) -> list[str]:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return []
    return [
        item.get("name", "")
        for item in data.get("models", [])
        if item.get("name")
    ]


def _build_pipeline_command(payload: dict) -> list[str]:
    command = [str(PYTHON_EXECUTABLE), "pipeline.py"]

    pitch = payload.get("pitch") or ""
    repo = payload.get("repo") or ""
    patents = payload.get("patents") or ""

    if pitch:
        command.extend(["--pitch", pitch])
    if repo:
        command.extend(["--repo", repo])
    if patents:
        command.extend(["--patents", patents])

    for flag in ("start_stage", "end_stage", "max_hops", "max_questions"):
        value = payload.get(flag)
        if value not in (None, ""):
            command.extend(["--" + flag.replace("_", "-"), str(value)])

    for flag in ("fusion_threshold", "resolver_threshold"):
        value = payload.get(flag)
        if value not in (None, ""):
            command.extend(["--" + flag.replace("_", "-"), str(value)])

    provider = payload.get("provider") or DEFAULT_PROVIDER
    command.extend(["--provider", provider])

    model = payload.get("model") or ""
    if provider == "ollama" and not model:
        models = _ollama_models()
        if DEFAULT_OLLAMA_MODEL in models:
            model = DEFAULT_OLLAMA_MODEL
        elif models:
            model = models[0]
        else:
            model = DEFAULT_OLLAMA_MODEL
    if model:
        command.extend(["--model", model])

    ground_truth = payload.get("ground_truth") or ""
    if ground_truth:
        command.extend(["--ground-truth", ground_truth])

    if payload.get("dry_run", False):
        command.append("--dry-run")

    return command


class Handler(BaseHTTPRequestHandler):
    server_version = "ChainCheckWeb/1.0"

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send_json(self, data: object, status: int = 200) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        content_type = "text/html; charset=utf-8"
        if path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif path.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._send_file(STATIC_DIR / "index.html")
            return
        if path.startswith("/static/"):
            self._send_file(STATIC_DIR / path.removeprefix("/static/"))
            return
        if path == "/api/project":
            self._send_json({
                "root": str(PROJECT_ROOT),
                "runtime": _runtime_info(),
                "defaults": _default_inputs(),
                "summary": _processed_summary(),
                "artifacts": {
                    key: {"file": filename, **_file_info(PROCESSED_DIR / filename)}
                    for key, filename in ARTIFACTS.items()
                },
            })
            return
        if path == "/api/job":
            self._send_json(JOB.snapshot())
            return
        if path == "/api/artifact":
            name = parse_qs(parsed.query).get("name", [""])[0]
            filename = ARTIFACTS.get(name)
            if not filename:
                self._send_json({"error": "Unknown artifact"}, status=404)
                return
            self._send_json(_read_json(PROCESSED_DIR / filename))
            return
        if path == "/api/questions":
            questions = _read_json(PROCESSED_DIR / "questions.json")
            templates = _read_json(PROCESSED_DIR / "due_diligence_questions.json")
            risks = _read_json(PROCESSED_DIR / "vc_risk_report.json")
            self._send_json({
                "questions": questions.get("questions", []),
                "template_questions": templates.get("questions", []),
                "risks": risks.get("identified_risks", []),
            })
            return

        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=400)
            return

        if parsed.path == "/api/run":
            command = _build_pipeline_command(payload)
            if not JOB.start(command):
                self._send_json({"error": "Pipeline is already running"}, status=409)
                return
            self._send_json({"started": True, "command": command})
            return

        self.send_error(404)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"ChainCheck frontend running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
