#!/usr/bin/env python3
"""Minimal OpenAI-compatible proxy backed by `codex exec`.

This is intentionally a thin chat-completions shim for local experiments.
It starts a fresh Codex CLI process per request and returns the last assistant
message as an OpenAI-style response.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4321
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_CODEX_HOME = Path(
    os.environ.get("CODEX_HOME", Path.home() / "mac_workspace" / ".codex-home")
)


SWE_LANCER_PROXY_PROMPT = """You are the model backend for an OpenAI-compatible Chat Completions API proxy.
Your job is to produce exactly the assistant message content that should be returned to the caller.

Important execution boundary:
- Do not inspect, read, modify, or create files on this host machine.
- Do not run shell commands yourself.
- Do not use Codex tools even if they are available.
- The caller may be SWE-Lancer's SimpleAgentSolver. In that setup, the caller executes Python code blocks from your reply inside the SWE-Lancer task container with `python -c`.
- Therefore, when the conversation asks you to interact with the repository, filesystem, tests, browser, or `/app/expensify`, return exactly one self-contained fenced Python code block.
- Do not claim that you already ran the code. The caller will run it and send back the output in the next message.
- When the task is complete, return exactly `DONE` and nothing else.
- If the conversation is a simple smoke test or ordinary chat that does not ask you to operate on the SWE-Lancer environment, answer normally.

Return only the assistant message content. Do not include API JSON, logs, or commentary about this proxy.
"""


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _error_response(handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
    _json_response(
        handler,
        status,
        {
            "error": {
                "message": message,
                "type": "codex_proxy_error",
                "param": None,
                "code": None,
            }
        },
    )


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _format_messages(messages: list[dict[str, Any]]) -> str:
    formatted: list[str] = []
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role", "user"))
        content = _message_content_to_text(message.get("content"))
        formatted.append(f"--- message {index}: {role} ---\n{content}")
    return "\n\n".join(formatted)


def build_codex_prompt(request_json: dict[str, Any]) -> str:
    messages = request_json.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("request must include a non-empty messages array")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")

    requested_model = request_json.get("model", DEFAULT_MODEL)
    conversation = _format_messages(messages)
    return (
        f"{SWE_LANCER_PROXY_PROMPT}\n"
        f"Requested chat model: {requested_model}\n\n"
        f"Conversation follows:\n\n"
        f"{conversation}\n\n"
        f"--- end conversation ---\n"
        f"Return the assistant message content now."
    )


def run_codex(
    prompt: str,
    model: str,
    workdir: Path,
    timeout_seconds: int,
    codex_home: Path | None,
) -> str:
    codex_bin = os.environ.get("CODEX_BIN", "codex")
    with tempfile.NamedTemporaryFile(prefix="codex-openai-proxy-", suffix=".txt", delete=False) as output_file:
        output_path = Path(output_file.name)

    cmd = [
        codex_bin,
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "-C",
        str(workdir),
        "-m",
        model,
        "-o",
        str(output_path),
        prompt,
    ]

    try:
        started = time.monotonic()
        env = os.environ.copy()
        if codex_home is not None:
            env["CODEX_HOME"] = str(codex_home)
        print(
            "starting codex exec "
            f"model={model} prompt_chars={len(prompt)} timeout={timeout_seconds}s "
            f"codex_home={env.get('CODEX_HOME', '<unset>')}",
            file=sys.stderr,
            flush=True,
        )
        completed = subprocess.run(
            cmd,
            cwd=workdir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        elapsed = time.monotonic() - started
        print(
            f"finished codex exec status={completed.returncode} elapsed={elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            details = stderr or stdout or f"codex exited with status {completed.returncode}"
            raise RuntimeError(details)

        if output_path.exists():
            content = output_path.read_text(encoding="utf-8").strip()
            if content:
                return content

        stdout = completed.stdout.strip()
        if stdout:
            return stdout
        raise RuntimeError("codex completed but produced no output")
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"codex exec timed out after {timeout_seconds} seconds") from exc
    finally:
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass


def chat_completion_response(content: str, model: str) -> dict[str, Any]:
    created = int(time.time())
    return {
        "id": f"chatcmpl-codex-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


class CodexProxyHandler(BaseHTTPRequestHandler):
    server: "CodexProxyServer"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            _json_response(
                self,
                200,
                {
                    "ok": True,
                    "model": self.server.codex_model,
                    "codex_home": str(self.server.codex_home)
                    if self.server.codex_home is not None
                    else None,
                },
            )
            return
        if path in {"/v1/models", "/models"}:
            _json_response(
                self,
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.codex_model,
                            "object": "model",
                            "created": 0,
                            "owned_by": "codex-openai-proxy",
                        }
                    ],
                },
            )
            return
        _error_response(self, 404, f"unknown endpoint: {path}")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/v1/chat/completions", "/chat/completions"}:
            _error_response(self, 404, f"unknown endpoint: {path}")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            request_json = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            _error_response(self, 400, f"invalid JSON: {exc}")
            return

        if request_json.get("stream"):
            _error_response(self, 400, "stream=true is not supported by this proxy")
            return

        try:
            prompt = build_codex_prompt(request_json)
            content = run_codex(
                prompt=prompt,
                model=self.server.codex_model,
                workdir=self.server.workdir,
                timeout_seconds=self.server.timeout_seconds,
                codex_home=self.server.codex_home,
            )
        except Exception as exc:
            _error_response(self, 500, str(exc))
            return

        _json_response(self, 200, chat_completion_response(content, self.server.codex_model))

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(
            f"{self.log_date_time_string()} {self.client_address[0]} {format % args}\n"
        )


class CodexProxyServer(ThreadingHTTPServer):
    codex_model: str
    workdir: Path
    timeout_seconds: int
    codex_home: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("CODEX_PROXY_HOST", DEFAULT_HOST))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CODEX_PROXY_PORT", str(DEFAULT_PORT))),
    )
    parser.add_argument("--model", default=os.environ.get("CODEX_PROXY_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path(os.environ.get("CODEX_PROXY_WORKDIR", os.getcwd())),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(os.environ.get("CODEX_PROXY_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))),
    )
    parser.add_argument(
        "--codex-home",
        type=Path,
        default=Path(os.environ["CODEX_HOME"])
        if "CODEX_HOME" in os.environ
        else DEFAULT_CODEX_HOME,
        help="CODEX_HOME to use for the child `codex exec` process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workdir = args.workdir.expanduser().resolve()
    server = CodexProxyServer((args.host, args.port), CodexProxyHandler)
    server.codex_model = args.model
    server.workdir = workdir
    server.timeout_seconds = args.timeout_seconds
    server.codex_home = args.codex_home.expanduser().resolve() if args.codex_home else None

    print(
        f"codex-openai-proxy listening on http://{args.host}:{args.port}/v1 "
        f"model={args.model} workdir={workdir} codex_home={server.codex_home}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down codex-openai-proxy", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
