#!/usr/bin/env python3
"""Colab-side helper for costsmart-rag local tiers (see docs/colab-handoff.md).

Two subcommands:

* ``serve`` -- run in Colab: loads a pinned local model with transformers
  and exposes the Ollama-compatible ``POST /api/generate`` endpoint the
  repo's ``ColabClient`` expects. Needs torch + transformers in the session.
* ``check`` -- run anywhere (stdlib only): probe a served endpoint with one
  tiny ``/api/generate`` call and report latency/tokens. This is how the
  cloud side verifies the captain-connected session is attached.

Examples:
    # in Colab
    python colab_local_tier.py serve --model Qwen/Qwen2.5-1.5B-Instruct --port 8000
    # cloud side
    python scripts/colab_local_tier.py check --base-url https://<tunnel-url>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def cmd_check(base_url: str, model: str, prompt: str, timeout_s: float) -> int:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001 - report any probe failure plainly
        print(f"check FAILED: {exc}", file=sys.stderr)
        return 1
    dt = time.monotonic() - t0
    text = payload.get("response", "")
    print(f"check OK: {dt * 1000:.0f}ms response={text[:120]!r} "
          f"eval_count={payload.get('eval_count', 0)}")
    return 0


def cmd_serve(model: str, port: int) -> int:
    try:
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError:
        print("serve needs torch + transformers in the Colab session: "
              "pip install torch transformers hf_xet", file=sys.stderr)
        return 2
    from http.server import BaseHTTPRequestHandler, HTTPServer

    print(f"loading {model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model)
    lm = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    lm.eval()
    print(f"serving {model} on :{port} (device: "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'})", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quieter logs
            pass

        def _send(self, payload: dict, status: int = 200) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/api/generate":
                self._send({"error": "want POST /api/generate"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, OSError):
                self._send({"error": "bad JSON"}, 400)
                return
            prompt = str(req.get("prompt", ""))
            inputs = tok(prompt, return_tensors="pt").to(lm.device)
            t0 = time.monotonic()
            with torch.no_grad():
                out = lm.generate(**inputs, max_new_tokens=256)
            gen = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            self._send({"response": gen,
                        "eval_count": len(tok.encode(gen)),
                        "latency_s": time.monotonic() - t0,
                        "model": req.get("model", model)})

    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Colab local-tier helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="probe a served endpoint (stdlib only)")
    p_check.add_argument("--base-url", required=True)
    p_check.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p_check.add_argument("--prompt", default="Say OK.")
    p_check.add_argument("--timeout-s", type=float, default=120.0)

    p_serve = sub.add_parser("serve", help="serve a local model (run in Colab)")
    p_serve.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    if args.cmd == "check":
        return cmd_check(args.base_url, args.model, args.prompt, args.timeout_s)
    return cmd_serve(args.model, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
