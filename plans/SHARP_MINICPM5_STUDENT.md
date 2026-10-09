# Sharp-MiniCPM5 student candidate

> **Retired (owner decision 2026-10-10):** `scripts/serve_minicpm.sh` were removed with the old candidate/teacher lines; see plans/LEGACY_CODE_REMOVAL.md. Restore from git history (`git log --diff-filter=D -- PATH`) if ever needed.

## Identity and acquisition

The intended publisher has `peculiar-ragdoll/Sharp-MiniCPM5-2B-GGUF`, not a
Sharp-MiniCPM2.5 release. Base: `openbmb/MiniCPM5-2B`, a 2.5B **dense** Llama
architecture, 42 full-attention layers, 16 query /2 KV heads, untied embeddings,
131,072 native context. Smaller weights suggest cheaper QLoRA than Spark, but
training activation memory and quality still require measurement.

- Quant repository revision: `040713a6c4da5e58e7512e8e483d315caef41b73`.
- Selected file: `Sharp-MiniCPM5-2B-Q4_K_XL.gguf`,1,595,727,200 bytes.
- Expected SHA256: `1185182d80019b86415de409584c2843dd43ae0b6211040fd363a5759cae99ed`.
- Local directory: `models/candidates/sharp-minicpm5-2b/`.
- Base tokenizer revision: `f97400052a43d642bbc6e9975e2397e3ae6a6b52`.
- Acquisition/verification artifacts: `runs/sharp-minicpm5-discovery/`.

Primary references: [quant/model card](https://huggingface.co/peculiar-ragdoll/Sharp-MiniCPM5-2B-GGUF),
[base config](https://huggingface.co/openbmb/MiniCPM5-2B/blob/f97400052a43d642bbc6e9975e2397e3ae6a6b52/config.json).
The Sharp release is quantization plus a template, not separate fine-tuned base
weights. Future training should use the original HF base in NF4/BF16 and our
verified template; the GGUF is for inference evaluation. Do not start training.
Suggested standard Llama LoRA modules: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`;
Spark's fused projection names are unsuitable here.

## Template and format contract

The GGUF metadata's embedded template matches the publisher's standalone Jinja.
Preserve that original in the candidate directory. Our reproducible local version:
`models/templates/Sharp-MiniCPM5-2B.jinja`, with companion kwargs JSON.

1. Default `terse=false`, avoiding an added house prompt conflicting with NatLang.
2. Default `enable_thinking=true` in both rendering and serving.
3. Retain native XML `<function name="..."><param name="...">` calls; let
   llama.cpp parse these to OpenAI `tool_calls`. Keep native tool instructions;
   do not suppress them for NatLang's standard function schema surface.
4. Local bug fix: content-block user queries also reset stale reasoning.
5. Local bug fix: a bare `]]>` triggers CDATA wrapping and splitting.
6. Training renderer normalizes JSON argument strings to mappings, already required
   by the original template. Preserve nested arguments and reasoning fields.
7. Training end token **`<|im_end|>`**, not nominal tokenizer EOS `</s>`.
   Base generation config includes both EOS IDs1 and130073. Do not append BOS
   externally: the template already inserts `<s>` and tokenizer add_bos is false.
8. Renderer now closes the real target directly, without a dummy following user
   that caused MiniCPM to discard target reasoning. Renderer version2 invalidates
   cache reuse; previously published static datasets are unchanged.

Before rendering training rows, copy the local patched Jinja to the downloaded
base tokenizer's `chat_template.jinja` (acquisition prepares this local bundle).
Use `scripts/render_training_corpus.py --model models/candidates/sharp-minicpm5-2b/tokenizer
--end-token '<|im_end|>'` with the desired input/output arguments. This is tokenizer
rendering only, not student training or a complete training-weight bundle.

## Isolated deployment

`scripts/serve_minicpm.sh --cpu` serves on localhost8082, one slot,4096 context,
3GiB container RAM/two CPUs, zero GPU layers, cached official llama.cpp image.
It uses `temp=1,top_p=.95,top_k=20,min_p=0` and q8_0 KV; **min_p=0** is the
publisher's repetition recommendation. No dynamic image/model lookup.

`scripts/serve_minicpm.sh --gpu` requires Bonsai to be paused at a case boundary
and its server stopped first; launcher refuses while `natlang-bonsai` runs.
Use `NATLANG_MINICPM_CTX=16384` for the later GPU pilot. These are total server
context sizes. Stop the isolated instance with `docker stop natlang-minicpm`.
Bonsai generation must continue during download and CPU compatibility checks.

## Validation and remaining work

- Local Jinja probes pass: text blocks, trailing interleaved content, nested JSON,
  CDATA split, bare terminator, stale reasoning removal, thinking prefix.
- GGUF header verified architecture and embedded-template identity.
- Renderer tests12/12 pass, including real MiniCPM Jinja/tool arguments/reasoning
  mask regression. Existing Spark nonthinking-profile probe passes.
- Full weight download/checksum, actual llama.cpp native tool parsing, and CPU
  smoke inference remain pending until acquisition completes. GPU benchmark and
  comparative reducer evaluation are subsequent work; no quality claim yet.


## Persistent acquisition state (20:15 UTC)

Large-file HTTP transfer was very slow and returned truncated ranges. Abandoned
both the sequential curl attempt and the temporary ranged downloader; do not
resume their `.part` as a sequential prefix (it contains holes). Only the final
HF/Xet file after SHA verification is usable. A partial inspection established
header/template identity, not tensor integrity.

Active acquisition container `natlang-minicpm-download` uses the already installed
training image's HF/Xet downloader (no GPU), fixed8 download concurrency,
2GiB RAM/two CPUs. No extra package installation was needed; the attempted host
pip installation was stopped. Log:`runs/sharp-minicpm5-discovery/xet-download-fixed.log`.

The durable `finish-acquisition.py` process logs to `finish.log` and independently
checks final model size/hash before CPU smoke. Status in `completion.json` is
`waiting_for_download`, `verifying_checksum`, `cpu_smoke`, `ready`, or `blocked`.
CPU smoke checks forced native tool-call parsing only, not reasoning quality or
NatLang reducer performance. Inspect `cpu-request.json`, `cpu-response.json`, and
`cpu-server.log` for actual outcomes. The worker stops only its isolated CPU
server; it leaves Bonsai running. On shutdown, explicitly stop the download
container and completion worker alongside generation; nothing here should be
mistaken for a finished benchmark.

The actual HF tokenizer loaded locally and rendered our patched Jinja, producing
one BOS and assistant terminator ID130073; teacher reasoning and XML tool-call
arguments passed rendering. Its nominal EOS remains `</s>` and must not be used
as the assistant turn delimiter in the SFT renderer.


## User-requested evaluation queued (about20:30 UTC)

The durable `runs/minicpm-eval-20260929/evaluate.py` worker PID14970 waits for
verified acquisition and CPU compatibility. It checks exact Bonsai supervisor
identity/start time; any change blocks the automatic swap. At a journaled case
boundary it preserves pending checkpoints, pauses the exact supervisor, waits
for children to exit, stops Bonsai's server, and uses the GPU for at most30
minutes. It restores the identical v39/queue-v31/journal, four requests, four
server slots/6GiB RAM/1,536MiB cache, recording the replacement supervisor PID.
Inspect `state.json` and `worker.log` before operating generation; avoid duplicate
manual launches. `report.json` will contain scores and missing-case denominators.

-24 held-out cases from shard-test across12 families: sufficiency, exact filters,
 multi-input judgment, structured extraction, paginated argmax/snapshots,
 directory criteria and reducer application, proof checking, union targets,
 stateful dates, parallel labels.
- All24 reference replays pass after reviewed contract modernization/externalization;
 original gold and source IDs preserved. Selection/reference evidence saved.
- Actor-routing pair excluded: one reference tried a flooded bridge and the
 pair-group checker complained about identical expected results. No model
 failure inferred from this broken reference; source/reference repair remains separate.
-13 application probes cover classification, ambiguous/missing evidence, urgency,
 contact extraction, arithmetic, child calls/sarcasm, slots, directory archive,
 malformed JSON handling. Blocker probes require relevant explicit model status,
 not an arbitrary runtime/transport exception.
- Frozen v39 runtime; greedy temperature0, root seed42, one global request,
 16K context,20 held-out turns,128 requests,120-second per-case wall cap.
 Application task limits and outer150-second cap prevent unbounded loops.
- Student collection role and separate evaluation artifacts; no admission into
 training. Single-seed pilot, not a matched Spark comparison or a training run.

Stopping the evaluator during a swap triggers server cleanup and Bonsai restore;
for whole-machine shutdown stop it first and then stop the restored supervisor.
Stop the acquisition completion worker too, so it cannot start its CPU smoke.


## Authenticated recovery (20:52 UTC)

Xet reconstruction failed with response-body decoding error. Acquisition and
queued evaluation watchers exited without pausing Bonsai. User then configured
local HF authentication. The downloader's authenticated API identity check
succeeds; token is read from a read-only mounted file, not logged or embedded in
commands. No evidence yet that the token improves payload bandwidth: previous
failure was a decoding error, not an HTTP429 quota response.

New `download-authenticated.py` uses authenticated HF metadata/resolve requests,
then ranged requests to the signed CDN URL without forwarding the HF bearer
credential to the CDN. Eight connections,1MiB chunks, durable fsynced range
journal, retry delay/backoff, and final exact size/SHA verification. Retains
completed ranges from earlier transport; never treat sparse file length as bytes
received. `download-state.json` gives completed bytes and session transfer rate.
Downloader container is detached and bounded to1GiB/two CPUs. Old Xet logs and
blocked statuses preserved. Current completion/evaluation watcher PIDs16040 and
16053; BonsaiPID8981 unchanged. These supersede earlier watcher PIDs.

Initial authenticated snapshot71,761,920/1,595,727,200 bytes (4.5%),~0.20MB/s,
zero retries. Evaluation remains pending real weight checksum and CPU smoke.
