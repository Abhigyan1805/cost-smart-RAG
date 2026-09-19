# Colab handoff: local tiers (costpilot-06)

Local model tiers run in **Google Colab, not on this machine**. The captain
connects the Colab session on request. This page is the exact handoff: what
to run in Colab, what to set cloud-side, and how to verify the attach.

Pinned local models (`config/models.yaml`):

| Tier | Model ID | Served as |
|---|---|---|
| local-small | `Qwen/Qwen2.5-1.5B-Instruct` | route L0/L1 |
| local-medium | `Qwen/Qwen2.5-7B-Instruct` | route C0 |

## Step 1 — in Colab (captain or whoever holds the session)

1. Open a Colab notebook with a GPU runtime (T4 is enough for local-small;
   L4 or better for local-medium).
2. Copy `scripts/colab_local_tier.py` from this repo into the Colab session
   (upload the file or `!curl` it from the repo raw URL).
3. Install + serve:
   ```sh
   pip install torch transformers hf_xet  # once per session
   python colab_local_tier.py serve --model Qwen/Qwen2.5-1.5B-Instruct --port 8000
   ```
   For local-medium, run a second cell/process with
   `--model Qwen/Qwen2.5-7B-Instruct --port 8001` (needs a bigger GPU).
4. Expose the port publicly (one of):
   - `ngrok http 8000` (print the `https://...` URL), or
   - Colab's built-in public serving / a reverse tunnel of your choice.
5. Note the public base URL, e.g. `https://abcd1234.ngrok.io`.

The server speaks the Ollama-compatible contract the cloud side expects:

- `POST {base}/api/generate` with JSON
  `{"model": "<id>", "prompt": "<text>", "stream": false}`
- responds with JSON `{"response": "<text>", "eval_count": <tokens>}`.

## Step 2 — cloud-side attach (no code changes)

```sh
export COSTSMART_COLAB_ENDPOINT="https://<your-tunnel-url>"   # required
export COSTSMART_COLAB_TOKEN="..."                            # only if the tunnel needs a bearer token
python scripts/colab_local_tier.py check --base-url "$COSTSMART_COLAB_ENDPOINT"
```

`check` is stdlib-only: it POSTs one tiny `/api/generate` probe and prints
latency + token counts. Exit 0 means the session is attached.

## Step 3 — verify through the repo client

```sh
PYTHONPATH=src python -c "
from costsmart.models.registry import get_client
c = get_client('local-small')   # -> ColabClient
print(type(c).__name__)
r = c.generate('Say OK.')
print(r.text[:200], r.tokens, round(r.latency_s, 2))
"
```

`ColabClient.generate()` reads `COSTSMART_COLAB_ENDPOINT` /
`COSTSMART_COLAB_TOKEN` and POSTs to `/api/generate`. With the env vars
unset it raises a `RuntimeError` pointing back to this page (never a silent
fallback, never a local-daemon assumption).

## Step 4 — re-run the pilot's local tiers (after attach)

```sh
PYTHONPATH=src python -m costsmart.eval.oracle_sweep --no-limit \
  --db results/costpilot-06/pilot.db \
  --config config/experiments/pilot-20.yaml
```

Rows are idempotent on `cache_key`, so re-running only fills what the stub
wrote before. Until this step runs, every local-tier number in
`results/costpilot-06/` is **stubbed/PRELIMINARY** and only the cloud-route
estimates use measured token counts x verified per-token prices.

## Cost note

Colab GPU time is amortized, not metered per token: record the session's GPU
type + hourly rate and queries served, then fill
`config/costs.yaml :: local_amortized` (method: plan section 5). The sweep
already logs `gpu_seconds` + `concurrency` per attempt for this.
