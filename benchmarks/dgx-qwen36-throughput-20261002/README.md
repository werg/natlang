# Qwen3.6 NVFP4 throughput benchmark

This benchmark is intentionally outside `runs/` and is not training-discoverable. The four captured request fixtures contain only exact prompt contexts and tool schemas from admitted NVFP4 teacher trajectories; they omit each captured response, source golds, and references. Fixture rows identify source result and request hashes for lineage. Treat these files as internal prompt data.

The stdlib Python client measures streaming time to first generated delta, p50/p95 TTFT and total latency, completion count/failures, OOM signatures, server reported prompt/completion usage, and aggregate prompt/output tokens per second. It never writes generated response content, tool arguments, auth values, raw exception strings, or server error bodies. It checks the pinned model ID at `/v1/models` and uses `/tokenize` to calibrate the synthetic input to exactly 1,500 tokens, and re-tokenizes captured inputs under the benchmark request settings.

## Quick pre-switch baseline

Copy only `benchmark_client.py` to DGX if running the synthetic workload first. With the current server at port 8082:

```sh
python3 /tmp/benchmark_client.py \
  --endpoint http://127.0.0.1:8082 \
  --concurrencies 16 \
  --requests-per-concurrency 2 \
  --workloads synthetic \
  --outdir /tmp/qwen-throughput-results
```

That sends 32 requests at concurrency 16, each with 1,500 measured input tokens and a 400-token output cap. `--workloads synthetic` does not require the captured fixtures. Results include telemetry only.

## Full sweep

Copy the whole benchmark directory to a scratch path on DGX, then run:

```sh
python3 /tmp/dgx-qwen36-throughput-20261002/benchmark_client.py \
  --endpoint http://127.0.0.1:8082 \
  --concurrencies 16,32,64,128,256 \
  --requests-per-concurrency 2 \
  --workloads synthetic,captured \
  --output-tokens 400 \
  --outdir /tmp/qwen-throughput-results
```

Each concurrency/workload cohort uses at least two waves. The synthetic cohort is 1,500 measured input tokens; the captured cohort rotates four teacher contexts measured at 5,201, 7,499, 10,501, and 15,709 input tokens with 6–22 messages and five tools. Stop-on-OOM is enabled by default; a cohort with an OOM signal stops the remaining sweep and records telemetry. No request is retried, so failures remain visible.

The root executed the sweep on2026-10-02; results are under `runs/dgx-throughput-20261002/results-upgraded/`. Re-running this client sends requests when invoked. It does not restart or configure vLLM. Output token throughput requires server streamed usage (`include_usage`); if unavailable, it reports the completion count but leaves aggregate output tokens per second unset rather than estimating from characters.

## Existing throughput context

Before the planned settings change, the root-run concurrency-16 synthetic baseline completed 32 requests with 1,500 measured input and 400 output tokens each: 12,800 output tokens in 37.3157 seconds, or 343.019 aggregate output tokens/s, with zero failures. The original TTFT values were null because the observer missed Qwen's `delta.reasoning` field; the client now detects it.

The active v32 real-teacher assignment provided a separate partial sample before its cutoff: 23 finished 16-case batches (368 cases) over 1,114.88 seconds, 7,410,424 input tokens and 226,529 output tokens. That is 6,646.8 prompt tokens/s and 203.2 output tokens/s at aggregate model concurrency 16, excluding startup and unfinished batches. These real multi-turn/tool cases are not directly comparable with the synthetic baseline. Full details and evidence hashes are in `comparison-context.json`.

The deployed post-restart profile: keep the pinned `eugr0.29.1rc` image, set GPU memory utilization to 0.65, max sequences to 256, and model length to 65,536, with root's remaining approved post settings. The client itself does not change or restart the server.


## Observed sweep2026-10-02

| Concurrency | Synthetic outputtok/s | Captured outputtok/s | Failures/OOM |
|---:|---:|---:|---:|
|32|444.5|336.9|0/0|
|64|540.3|458.6|0/0|
|128|634.8|557.1|0/0|
|256|699.2|654.5|0/0|

All synthetic completions reached the400-token cap with reasoning-only output. The captured workload reuses four contexts, so prefix caching benefits these results; under this benchmark's actual template, average captured input was about7,929 tokens. Captured256 TTFTp95 was21.17seconds and latency p95 was148.64seconds. These measurements do not establish correct model answers or whole-trajectory accepted throughput.

The recipe's claimed2835tok/s was not reproduced. We kept the compatible pinned `eugr` image rather than switching to an unpinned official nightly; FP8 KV, Marlin, prefix caching and chunked-prefill are enabled. Production starts128 requests and increases to256 after two complete-accounting batches.
