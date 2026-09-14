#!/usr/bin/env python3
"""LAN-facing OpenAI-compatible shim in front of FastLLM / v0.2."""
import http.client
import http.server
import json
import os
import socketserver
import sys
import time
from urllib.parse import urlsplit

BACKEND = os.environ.get("COMPAT_BACKEND", "http://127.0.0.1:8001")
PORT = int(os.environ.get("COMPAT_PORT", "8000"))
# Short names shown in Chatbox. Old long *-think-* aliases still work.
THINK_PUBLISH = (
    ("qwen3.8-27b-think-low", "low"),
    ("qwen3.8-27b-think-medium", "medium"),
    ("qwen3.8-27b-think-xhigh", "xhigh"),
)
THINK_BY_NAME = {name: effort for name, effort in THINK_PUBLISH}
LEGACY_SUFFIX = (
    ("-think-low", "low"),
    ("-think-medium", "medium"),
    ("-think-xhigh", "xhigh"),
    ("-think", "medium"),
)
_BACKEND_MODEL = None
HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
CORS_HOP = {
    "access-control-allow-origin",
    "access-control-allow-credentials",
    "access-control-allow-methods",
    "access-control-allow-headers",
    "access-control-max-age",
    "access-control-expose-headers",
}


def backend_addr():
    parsed = urlsplit(BACKEND)
    return parsed.hostname or "127.0.0.1", parsed.port or 80


def sanitize_models(raw):
    data = json.loads(raw)
    items = data.get("data") if isinstance(data, dict) else None
    if not items:
        return raw
    created = int(time.time())
    slim = []
    seen = set()
    for item in items:
        mid = item.get("id")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        owned = item.get("owned_by") or "fastllm"
        slim.append(
            {
                "id": mid,
                "object": "model",
                "created": created,
                "owned_by": owned,
            }
        )
        if str(mid) not in THINK_BY_NAME and split_think_alias(str(mid))[0] is None:
            remember_backend_model(str(mid))
            for alias, _effort in THINK_PUBLISH:
                if alias not in seen:
                    seen.add(alias)
                    slim.append(
                        {
                            "id": alias,
                            "object": "model",
                            "created": created,
                            "owned_by": owned,
                        }
                    )
    return json.dumps({"object": "list", "data": slim}, ensure_ascii=False).encode("utf-8")


def remember_backend_model(model):
    global _BACKEND_MODEL
    if model and model not in THINK_BY_NAME:
        _BACKEND_MODEL = model


def resolve_backend_model():
    if _BACKEND_MODEL:
        return _BACKEND_MODEL
    try:
        host, port = backend_addr()
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/v1/models")
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        data = json.loads(raw)
        for item in data.get("data") or []:
            mid = item.get("id")
            if mid and mid not in THINK_BY_NAME:
                remember_backend_model(str(mid))
                break
    except Exception:
        pass
    return _BACKEND_MODEL


def split_think_alias(model):
    if not isinstance(model, str):
        return None, None
    if model in THINK_BY_NAME:
        return _BACKEND_MODEL, THINK_BY_NAME[model]
    for suffix, effort in sorted(LEGACY_SUFFIX, key=lambda item: -len(item[0])):
        if model.endswith(suffix):
            real = model[: -len(suffix)]
            if real:
                return real, effort
    return None, None


def rewrite_think_request(body):
    """Map think aliases to the real served model and set reasoning_effort."""
    if not body:
        return body, None
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return body, None
    if not isinstance(data, dict):
        return body, None
    real, effort = split_think_alias(data.get("model"))
    if not effort:
        return body, None
    if not real:
        real = resolve_backend_model()
    if not real:
        return body, None
    data["model"] = real
    kwargs = data.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        kwargs = {}
    kwargs["enable_thinking"] = True
    kwargs["reasoning_effort"] = effort
    data["chat_template_kwargs"] = kwargs
    return json.dumps(data, ensure_ascii=False).encode("utf-8"), effort


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def cors(self):
        origin = self.headers.get("Origin") or "*"
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Credentials", "true")
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS",
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            self.headers.get("Access-Control-Request-Headers") or "*",
        )

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        self.handle_any()

    def do_POST(self):
        self.handle_any()

    def do_PUT(self):
        self.handle_any()

    def do_PATCH(self):
        self.handle_any()

    def do_DELETE(self):
        self.handle_any()

    def do_HEAD(self):
        self.handle_any()

    def mapped_path(self):
        raw_path = self.path.split("?", 1)[0]
        query = self.path[len(raw_path) :]
        if self.command == "POST" and raw_path in (
            "/models",
            "/v1/models",
            "/v1/model",
            "/chat/completions",
        ):
            return "/v1/chat/completions" + query, False
        if raw_path in ("/models", "/v1/model"):
            return "/v1/models" + query, True
        if raw_path == "/v1/models":
            return self.path, True
        return self.path, False

    def send_text(self, code, body, content_type="text/plain; charset=utf-8"):
        payload = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.cors()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def handle_any(self):
        raw_path = self.path.split("?", 1)[0]
        if raw_path == "/metrics":
            self.send_text(200, "# TYPE up gauge\nup 1\n")
            return
        if raw_path in ("/health", "/healthz", "/ready"):
            self.send_text(200, "ok\n")
            return

        path, is_models = self.mapped_path()
        if path != self.path:
            sys.stderr.write(
                "%s - rewrite %s %s -> %s\n"
                % (self.address_string(), self.command, self.path, path)
            )
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        host, port = backend_addr()
        headers = {}
        for key, value in self.headers.items():
            low = key.lower()
            if low in HOP or low == "host":
                continue
            headers[key] = value
        headers["Host"] = "%s:%s" % (host, port)
        chat_path = path.split("?", 1)[0]
        if self.command == "POST" and chat_path in (
            "/v1/chat/completions",
            "/chat/completions",
        ):
            body, effort = rewrite_think_request(body)
            if effort:
                sys.stderr.write(
                    "%s - think alias -> reasoning_effort=%s\n"
                    % (self.address_string(), effort)
                )
                headers["Content-Length"] = str(len(body))
                headers["Content-Type"] = "application/json"

        try:
            conn = http.client.HTTPConnection(host, port, timeout=3600)
            conn.request(self.command, path, body=body, headers=headers)
            resp = conn.getresponse()
        except Exception as exc:
            self.send_text(502, "backend unavailable: %s\n" % exc)
            return

        if is_models and self.command in ("GET", "HEAD"):
            raw = resp.read()
            try:
                raw = sanitize_models(raw)
            except Exception:
                pass
            self.send_text(resp.status, raw, "application/json")
            conn.close()
            return

        self.send_response(resp.status)
        for key, value in resp.getheaders():
            low = key.lower()
            if low in HOP or low in CORS_HOP:
                continue
            self.send_header(key, value)
        self.cors()
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            # Keep complete SSE lines, but never block waiting for a big buffer.
            # read(8192) waits until 8KB arrives, so the speed-test page sees
            # decode tokens in bursts and reports 2x output speed.
            buf = b""
            read1 = getattr(resp, "read1", None) or getattr(getattr(resp, "fp", None), "read1", None)
            while True:
                chunk = read1(4096) if read1 else resp.read(256)
                if not chunk:
                    break
                buf += chunk
                while True:
                    idx = buf.find(b"\n")
                    if idx < 0:
                        break
                    self.wfile.write(buf[: idx + 1])
                    self.wfile.flush()
                    buf = buf[idx + 1 :]
            if buf:
                self.wfile.write(buf)
                self.wfile.flush()
        conn.close()


def main():
    host, port = backend_addr()
    print(
        "compat-proxy listen=0.0.0.0:%s backend=%s:%s" % (PORT, host, port),
        flush=True,
    )
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
