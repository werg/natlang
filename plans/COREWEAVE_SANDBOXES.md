# CoreWeave serverless GPU sandboxes

Status 2026-10-12 (DGX). The owner has preview access, and the keys are not set yet. Everything below that needs no
key is built: the controller, the bootstrap, content-addressed bundles and tests (`scripts/coreweave/`). The first
commands to run once the key arrives are in "Bring-up checklist". The goal is to take training and eval pressure off
the DGX GB10, whose 121 GB unified memory is the bottleneck. Today the Mellum v4 run holds 98 GB of it, and the Mellum
base task reference (34 GB) waits for a pause window.

## 1. Facts from the documentation

All facts come from docs.coreweave.com (fetched 2026-10-12) unless a line says otherwise. "Unknown" means the docs
don't say.

| Topic | Fact | Source |
| - | - | - |
| GPU type (serverless) | One model only: **NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB** (GDDR7, compute capability 12.0, `sm_120`; needs CUDA ≥ 12.8). The `type` key is a filter, not a menu; leave it unset. | [gpu-sandboxes](https://docs.coreweave.com/products/sandboxes/gpu-sandboxes.md) |
| GPUs per sandbox | **Exactly 1 on serverless.** Multi-GPU (up to a full node) only on your own CKS cluster with a runner. | same |
| Host behind it | 8x RTX PRO 6000 nodes on PCIe (no NVLink), Intel Emerald Rapids 8562Y+, 128 vCPU, 1024 GB RAM (High Memory) or 512 GB (Standard), 7.68 TB local NVMe RAID1, 2x100GbE. Which node a sandbox lands on is not selectable. **x86_64.** | [rtxp6000-8x](https://docs.coreweave.com/platform/instances/gpu/rtxp6000-8x.md), [rtxp6000-8x-v2](https://docs.coreweave.com/platform/instances/gpu/rtxp6000-8x-v2.md) |
| GPU access | Private preview. The organization must be allowlisted, or creation fails with `CWSANDBOX_GPU_NOT_ALLOWED`. The TypeScript client cannot request GPUs; use Python `cwsandbox >= 1.14.2`. | gpu-sandboxes |
| Isolation | Serverless GPU sandboxes run in a GPU virtual machine (secure runtime). The platform provides the driver and `nvidia-smi`. GPU sandboxes start more slowly ("allow several minutes"). | gpu-sandboxes |
| Lifetime | `max_lifetime_seconds` up to **30 days**. The default is only **10 minutes** if unset. It **cannot be extended** after start. On expiry the Kubernetes Job is killed **immediately, with no grace period** and no distinguishable reason. | [long-running-sandboxes](https://docs.coreweave.com/products/sandboxes/long-running-sandboxes.md), [sandbox-configuration](https://docs.coreweave.com/products/sandboxes/client/guides/sandbox-configuration.md) |
| Stop | `stop(graceful_shutdown_seconds=…)` gives processes a grace period (client default 10 s; the cap for a normal stop is unknown; capped at 300 s with snapshot-on-stop). | [client ref Sandbox](https://docs.coreweave.com/products/sandboxes/client/ref/core/sandbox.md) |
| Suspend / resume | No live pause of a running process. A `PAUSED` status exists but has no documented user API. **File System Snapshots** (FSS) suspend and resume a *directory*: a scratch mount at a path is archived (tarball) to an org-scoped bucket on stop or mid-life and restored into a new sandbox, or forked into several. Sizes and limits are unknown (`CWSANDBOX_FSS_SIZE_EXCEEDED` exists). | [file-system-snapshots](https://docs.coreweave.com/products/sandboxes/file-system-snapshots.md) |
| Main process | When the main command exits with 0 the sandbox becomes `COMPLETED` and billing stops. **A nonzero exit restarts the container** (`RUNNING → RUNNING`). | [sandbox-status ref](https://docs.coreweave.com/products/sandboxes/client/ref/configuration/sandbox-status.md), lifecycle |
| Preemption | No preemption is documented for serverless sandboxes. The risk is lifetime expiry and capacity errors at create time (`SandboxResourceExhaustedError`, `CWSANDBOX_RUNNER_UNAVAILABLE`). | gpu-sandboxes, limits |
| Concurrency | The concurrent-sandbox quota is org-wide (default 5,000 for CoreWeave accounts) when enforcement is on. Create requests are also rate-limited (`CWSANDBOX_SERVERLESS_CREATE_THROTTLED`, HTTP 429). **GPU capacity is not guaranteed** by the quota. Our real GPU concurrency limit is unknown; ask the account team. | [limits-and-quotas](https://docs.coreweave.com/products/sandboxes/reference/limits-and-quotas.md) |
| Images | Any image the runner can pull. The default is `python:3.11`. Private registries take `image_pull_credentials` (`ImagePullCredentials(registry, store, name)`) resolved from a secret store (the documented store is `wandb`). Images must carry CUDA ≥ 12.8 and a framework built for `sm_120`. | sandbox-configuration, [containers ref](https://docs.coreweave.com/products/sandboxes/client/ref/workload/containers.md) |
| Resources | Serverless sets requests equal to limits (Guaranteed QoS). CPU and memory must be set explicitly next to a GPU. There is no disk field: the root filesystem is node-local ephemeral, and a dedicated path is a **scratch volume** (`ScratchVolumeOptions(name, mount_path, size)`, disk or memory). | gpu-sandboxes, [volumes](https://docs.coreweave.com/products/sandboxes/volumes.md) |
| Shared volumes | Registered Volumes are PVCs in *your* CKS cluster, so they are not available to plain serverless. Serverless has no shared filesystem. | volumes |
| Data in and out | (a) **CoreWeave AI Object Storage** (S3-compatible, endpoint `https://cwobject.com`, virtual-hosted addressing required) with **Workload Identity Federation**: the sandbox gets a token and the AWS SDK exchanges it for temporary credentials. You opt in per sandbox with `object_storage_access={buckets, permission}`. It needs a one-time admin setup and `boto3 >= 1.42.5`. (b) `read_file`/`write_file`/streaming file calls through the API, only for small files: the docs say use object storage for large data, and cwsandbox 1.19.0 has `DEFAULT_FILE_OPERATION_CAP_BYTES = 32 MiB`. (c) `mounted_files` (text, at start). | [object-storage-access](https://docs.coreweave.com/products/sandboxes/operations/object-storage-access.md), [file-operations](https://docs.coreweave.com/products/sandboxes/client/guides/file-operations.md) |
| Object storage from DGX/Pop | One CoreWeave API token serves both. It is exchanged at `https://api.coreweave.com/v1/cwobject/temporary-credentials/api-token` (`AWS_CONTAINER_CREDENTIALS_FULL_URI` + `AWS_CONTAINER_AUTHORIZATION_TOKEN="Bearer …"`). This needs `cwobject:CreateAccessKey` (Object Storage Admin or an org policy). `cwic` reads `COREWEAVE_API_TOKEN`. | [api-access-token exchange](https://docs.coreweave.com/products/storage/object-storage/auth-access/manage-access-keys/api-access-token.md) |
| Network | `NetworkOptions(deny_egress, egress=[EgressRule(dns_name/cidr/…)], …)`. The serverless default egress is the CoreWeave policy's, which the docs don't state (unknown whether pip, nodejs.org or nvcr.io are reachable; image pulls are platform-side). `NetworkService` is unimplemented. Ingress goes through `services`/public endpoints. | [network-options ref](https://docs.coreweave.com/products/sandboxes/client/ref/configuration/network-options.md), [protos](https://docs.coreweave.com/products/sandboxes/reference/protos.md) |
| Auth | `CWSANDBOX_API_KEY` = a CoreWeave API access token (console → Tokens) with the **Sandbox User** role (`SANDBOX_USER`). Alternatively `WANDB_API_KEY` with `AuthStrategy.WANDB` (serverless only, Forge billing). Admin actions (runners, WIF config) need `SANDBOX_ADMIN`. | [placement](https://docs.coreweave.com/products/sandboxes/placement.md), [control-plane-api](https://docs.coreweave.com/products/sandboxes/reference/control-plane-api.md) |
| API / SDK / CLI | Python `cwsandbox` (we use 1.19.0) is the supported surface: `Sandbox.run/list/from_id/stop/exec/stream_logs/read_file/write_file/snapshot`. Its CLI `cwsandbox` (`[cli]` extra) has `ls get exec logs stop files snapshots sh delete`, but **no create**. The data plane is `https://api.cwsandbox.com` (Connect/gRPC; REST e.g. `POST /v1beta2/sandboxes`, `/stop`, `/file-system-snapshots`); the control plane is `https://api.coreweave.com/v1/sandbox` (runners and policies, admin only). The protos are at `buf.build/coreweave/sandbox`. The published `openapi.yaml` link returns an HTML page. | [client](https://docs.coreweave.com/products/sandboxes/client.md), protos |
| Logs | `stream_logs(follow, tail_lines, since_time, timestamps)` returns the **main process** stdout/stderr only; `exec` output is separate. Fleet metrics (`cwsandbox_*`, PromQL) are in CoreWeave's managed Grafana. | [logging](https://docs.coreweave.com/products/sandboxes/client/guides/logging.md), [usage-metrics](https://docs.coreweave.com/products/sandboxes/operations/usage-metrics.md) |
| Sandbox env | The platform sets `CW_SANDBOX_ID`. The docs warn against putting credentials in `environment_variables`; use `secrets=[Secret(store, name, env_var)]`. | [environment-variables](https://docs.coreweave.com/products/sandboxes/environment-variables.md) |
| Price, CPU/memory | **$0.128 per CPU-core-hour and $0.0212 per GiB-hour of memory**, metered per millisecond on the **request**, not the use, from ready to terminated. Scheduling, image pull and cleanup are not billed. There is no idle timeout, and no internet egress fee. | [Forge pricing](https://coreweave.com/forge-pricing/inference-agents) |
| Price, GPU | "Private Preview", unpublished: billed per GPU-hour by model, rates from the account team. For reference only, the RTX PRO 6000 High-Memory 8-GPU instance is $20.00/h on demand ($2.50 per GPU-hour) and $11.01/h spot. | [Forge pricing](https://coreweave.com/forge-pricing/inference-agents), [pricing](https://www.coreweave.com/pricing) |
| Price, storage | AI Object Storage is $0.06 per GB-month hot, $0.03 warm and $0.015 cold, with no ingress, egress or request fees. FSS and volume storage are billed separately. | [pricing](https://www.coreweave.com/pricing) |

Where the docs and the client disagree, the installed client wins. cwsandbox 1.19.0 rejects `max_timeout_seconds`
with "removed in 1.x", although the configuration guide still lists it. It also reads list-form `mounted_files` as
`{mount_path, file_content}`, while the guide shows `{path, content}`. `cwctl.py plan` therefore prints the wire
request that the installed client would actually send, built offline.

We measured our own image on the DGX (2026-10-12). NGC `pytorch:26.09-py3` ships PyTorch 2.14.0a0 with CUDA 13.4,
Triton 3.8.0 and transformers 5.19.0, and its arch list includes `sm_120`. The NGC tag is multi-arch, so CoreWeave
pulls the x86_64 variant. Our local `natlang-neuralese:26.09` is **arm64** (the GB10 is aarch64), so it cannot run on
CoreWeave as built.

## 2. What would move, and what it needs

| Workload | Today | Inputs (size) | GPU memory | Fit on 1× RTX PRO 6000 |
| - | - | - | - | - |
| Mellum base task reference (heldv2, 16 cases) | Waits for the DGX pause window (34 GB, NATLANG_PILLAR item 0) | Model dir 23 GB, heads 124 MB, packet 0.3 MB, code 58 MB, built ts-host, node ≥ 22 | ~34 GB | Yes. **First real job.** |
| Task-success evals and base references (any line) | DGX ledger, medium | Same pattern; LFM inputs are small | ≤ 34 GB | Yes, and parallel per checkpoint |
| Mellum v4 direct SFT and trajectory (MuonSR, FP8 experts, QAT, nested family) | DGX, 98 GB ledger claim | Union r3 Mellum renders (program-header 6.3 GB, whole-trajectory 2.2 GB, full 3.3 GB), r2 program-header 2.0 GB (continued data), model 23 GB, heads, expert order, state `checkpoint.pt` **47 GB** | ~50 GB allocated, 68 GB reserved steady state, plus 11–12.5 GB peak (v4 train.jsonl) | Yes (96 GB). Host RAM is separate: request ≥ 128 Gi |
| E2E trainer arms, dragnets, ablations | Serialized on DGX | Export plus union subsets | Per arm | One sandbox per arm, in parallel |
| Data rendering (union twins, packets) | DGX/Pop CPU, low priority | Corpus snapshots | none | CPU sandboxes; only if rendering needs no teacher on :8082 (unreachable from CoreWeave) |
| Memetic and RL smokes | DGX | Small | Small to medium | Yes |
| LFM lines | Pop-owned | Small | Small | Pop's decision (brief, don't move) |

The union snapshots total 11 GB (r2), 24 GB (r3, both tokenizers) and 4.7 GB (r2 context recovery). A Mellum stage
reads only its tokenizer's twins. The corpus manifests (`training/corpus-manifests/<id>.json`, 1,377 of them) already
pin each file's sha256 and size, so bundling them needs no rehash.

## 3. What the trainers assume about the GB10, and the changes, ranked

From a code inventory of `training/neuralese` on 2026-10-12. File:line references are in the session record; the main
ones are below.

1. **Architecture: x86_64 plus `sm_120` instead of aarch64 plus `sm_121`.** No code change is needed. The image is
   either the public multi-arch NGC 26.09 with requirements installed at start (bootstrap), or an amd64 build of the
   same Dockerfile (`scripts/coreweave/build_image.sh`). `sm_120` and `sm_121` are the same Blackwell family, both
   with about 100 KB of shared memory per SM, so the Triton tiles measured on the GB10 (`maple/fused_moe.py`,
   `maple/expert_gemm.py` BM/BN/BK 128/128/64, 3 stages) stay valid. Retuning them is an optimization, not a blocker.
   The FP8 `triton-experts` backend needs SM ≥ 8.9 and has no `sm_121`-only path.
2. **Memory model: 96 GB discrete GPU plus separate host RAM, instead of 121 GB unified.** `direct_sft.execution_plan`
   sizes itself from `torch.cuda.mem_get_info` plus the reusable allocator cache, and from the ledger claim only when
   `NATLANG_LEDGER_UNIT` is set (`memory_policy.py`). Without a ledger it runs on device memory alone, and the
   page-cache release is skipped when the ledger script is missing. **No change.** The checkpoint load
   (`map_location=cpu`) needs host RAM: request 128 Gi for Mellum training and measure, because memory is billed on
   the request.
3. **Paths.** The defaults are absolute DGX paths (`common/paths.py`), and `recipe-plan.json`, staged inputs and
   heads checkpoints record absolute paths. For example, v4's `continue_from` is under `/mnt/external/…`, and the
   heads pin the base directory. **Fix (in the bootstrap): mirror roots.** `/home/werg` and `/mnt/external` become
   symlinks into the scratch volume, and bundles materialize files at their recorded paths. `NATLANG_REPO` is the same
   path as on DGX, so a moved run resumes with identical identities. No trainer change is needed.
4. **Checkpoint egress.** `checkpoint.pt` is 47 GB (the hard-linked `best-checkpoint.pt` costs nothing extra), and a
   49 GB `.checkpoint-space.reserve` sits next to it. Size scratch at ≥ 150 Gi per Mellum stage plus inputs (~60 GB):
   about 300 Gi. `bundle.py publish` skips `*.reserve` and the trainers' `*.pending` temp files, and deduplicates
   hard links by content. `bundle.py watch` runs beside the job and publishes each output directory whenever a
   checkpoint or report changes, so lifetime expiry, which kills without grace, loses at most one checkpoint
   interval (every 170 updates).
5. **Shutdown.** The trainers write their final state on SIGTERM, and v4 used `--stop-seconds 660`. `cwctl.py stop`
   defaults to `--graceful-shutdown-seconds 600` (`NATLANG_CW_STOP_GRACE`). The bootstrap forwards TERM to the job and
   publishes outputs after it exits. The documented cap for a normal stop's grace is unknown; verify it in the first
   training run.
6. **Exit status.** A nonzero main-process exit restarts the container, which would rerun bootstrap and job. The
   bootstrap therefore always exits 0. It records the real status in the log, in `$WORK/.coreweave/exit-status` and in
   `<output prefix>exit.json`.
7. **No multi-GPU in the trainers** (no torch.distributed, DDP or FSDP anywhere in `training/neuralese`), and
   `devices.py` hard-codes device 0. Serverless gives exactly one GPU anyway. **Parallelism is across sandboxes**:
   independent arms, evals and lines. DDP or FSDP becomes relevant only with a CKS cluster, which is a later and
   separate decision.
8. **torch.compile and inductor caches start cold in every sandbox** (Lion/AdaMini, flex attention, and Maple
   rms_norm/MoE when `NATLANG_MAPLE_COMPILE=1`). Measure the cost in the first run. If it matters, publish
   `TORCHINDUCTOR_CACHE_DIR` as a bundle, keyed by code sha and GPU.
9. **The memory ledger does not apply remotely.** A sandbox owns its GPU, so admission is CoreWeave's. The ledger
   keeps governing DGX. Remote jobs are tracked by `cwctl.py` records, the coord status line and sandbox tags.
10. **Task-success eval needs node ≥ 22 and a built ts-host.** `ts-host/dist` is gitignored, so it is not in the code
    archive. The bootstrap installs node from nodejs.org (checked against `SHASUMS256.txt`) and runs
    `npm ci && npm run build` when `NATLANG_CW_BUILD_TSHOST=1`. This needs egress to nodejs.org and npm (unknown;
    first smoke). The fallback is a prebuilt image that includes node.

## 4. Design

### Placement policy is a natlang program; mechanism is crisp

The decision "which job runs where" is policy, so it is a natural-language function, as AGENTS.md requires:

```
placeJob(job: JobDeclaration, fleet: FleetFacts) -> { where: "dgx" | "pop" | "coreweave", reason: string, resources?: … }
```

It follows these steps over named data, written for a small executor:

1. Read the job's declared needs: GPU memory, host memory, hours, inputs and whether they are already uploaded, and
   whether it needs the teacher on :8082 or Pop-only data.
2. Read the fleet facts. For DGX: the ledger's headroom and queue. For Pop: its status page. For CoreWeave: whether
   GPU access is live, the active sandboxes this machine launched, the GPU rate if quoted, and the budget left this
   week.
3. If the job needs something only one machine has (teacher, Pop-owned lineage, local-only data), place it there.
4. If it fits DGX headroom now without preempting training, prefer DGX: it costs nothing extra.
5. Otherwise, if it fits one RTX PRO 6000 (≤ 90 GB GPU) and its inputs can be bundled, place it on CoreWeave. Choose
   CPU and memory from its measured peak, not a guess, because the request is billed.
6. Otherwise, queue it on the DGX ledger and say what would unblock it.
7. Return the reason in one or two sentences. It is recorded in the job's `placement.reason`.

Crisp parts, each justified:

- **`scripts/coreweave/job.py`** is the job declaration (`natlang.coreweave-job/1`) and its exact translation into a
  `Sandbox.run` request. It is an exact verifier of what the policy produced: limits, serverless 1 GPU, lifetime
  ≤ 30 d, no credentials in env, a reason present.
- **`scripts/coreweave/cwctl.py`** is the CLI for the outside world (CoreWeave's API) and is the only module that
  creates sandboxes:
  - `doctor` and `plan` (the dry run: request, wire request built offline by the installed client, cost bound);
  - `launch`, `smoke`, `records` and `status-line`;
  - pass-through of `ls get exec logs stop files snapshots sh delete` to the official `cwsandbox` CLI, so there is one
    client implementation;
  - a local launch record (`~/.local/state/natlang/coreweave/sandboxes.jsonl`, `NATLANG_CW_STATE`);
  - sandbox tags `natlang`, `job-<id>` and `kind-<kind>`.
- **`scripts/coreweave/bundle.py`** handles exact durability and transfer: content-addressed upload, verified fetch,
  output publish and verify.
- **`scripts/coreweave/bootstrap.sh`** is in-sandbox plumbing: deps, mirror roots, code snapshot, inputs, optional
  ts-host, run, publish.
- `setup_client.sh` creates the client venv. `build_image.sh` builds the amd64 image.

**Built: the placement program, `applications/ops-placement/`** (its DECOMPOSITION.md comes first).

- `placeJob.nl` is six numbered steps over named facts. In order: a Pop-owned lineage goes to Pop; moving an active
  lineage goes to the owner (ask); a job that needs the teacher goes to DGX; a job that fits DGX headroom with no
  queue ahead goes to DGX; a job that fits one sandbox GPU, within the parallel limit and the budget, goes to
  CoreWeave; otherwise the DGX queue, naming the blocker.
- `scripts/coreweave/placement.py` gathers the facts: the DGX ledger headroom and queue, this machine's launch
  records, readiness (credential, bucket, a passed GPU smoke), and the budget from `training/coreweave-placement.json`.
  It also holds the crisp implementation of the same steps.
- It is pluggable: `crisp` (default), `nl` or `shadow`, through ts-host's `pluggable`, with an exact `refusal()`
  bound in index.ts and the bridge `ts-host/scripts/placement-decide.mjs`.
- `cwctl.py place JOB.json [--mode …] [--write]` records the served decision in the job, and `cwctl.py launch`
  refuses a job whose decision is not `coreweave`.
- The instructions are model-facing and unmeasured, recorded as debt in plans/MODEL_FACING_CHANGES.md.

**Coordination.** Every launch is recorded locally. `cwctl.py status-line` gives the line for this machine's
`coord.py status --set` page. Jobs that change shared lineages (a Mellum stage moving to CoreWeave) are briefed to
Pop as a `decision`. The machine that launches a sandbox owns it, the same rule as for local jobs.

### Environment: bootstrap first, image later

- **Phase 1 (now): bootstrap.** Use the public `nvcr.io/nvidia/pytorch:26.09-py3`. The pip install comes from
  `training/neuralese/docker/requirements.txt`, the same list the Dockerfile now installs, so there is one dependency
  list. It costs about a minute of billed start-up per sandbox (pull time is not billed).
- **Phase 2:** `NATLANG_CW_IMAGE_REPO=ghcr.io/<owner>/natlang-neuralese scripts/coreweave/build_image.sh` builds the
  amd64 image with buildx. On DGX this needs QEMU binfmt (not installed; the builder is arm64 only), or it can be built
  on Pop (x86_64). Then set `"image"` and `NATLANG_CW_IMAGE_READY=1` in the job. A public repository avoids pull
  credentials. A private one needs `image_pull_credentials` from a secret store (`wandb` store; requires W&B auth
  setup).

### Data path: one bucket, keyed by sha256

```
s3://<bucket>/
  cas/sha256/<2 hex>/<64 hex>        every input and output file and code snapshot, by content (deduplicated)
  bundles/<manifest sha256>.json     natlang.coreweave-bundle/1: [{path, sha256, bytes, key}], code {git_sha, sha256, key}
  outputs/<job id>/<dir>/latest.json pointer to the newest published output manifest of that directory
  outputs/<job id>/exit.json         the job's real exit status
```

- **Plan:** `bundle.py plan --recipe R --stage S --corpus ID --extra PATH --code-sha SHA -o m.json`. Recipe bindings
  and corpus manifests reuse their pinned sha256. Other files are hashed through the shared cached hasher
  (`natlang_neuralese.common.hashing`). The code is `git archive` of one commit, the same frozen-snapshot practice as
  the DGX launchers, so no GitHub credential enters a sandbox.
- **Upload:** `bundle.py upload m.json --bucket B`. Keys that already exist are skipped, so a re-upload is a no-op
  and the 23 GB model is uploaded once for every job.
- **Arrival:** the bootstrap verifies the code tarball's sha256. Then `bundle.py fetch` hashes each file before
  moving it into place, refuses any mismatch, and writes `.coreweave/receipts/<bundle>.json`. Copies are not
  admission: the recipe's pinned hashes still decide.
- **Credentials:**
  - In the sandbox: WIF, so no key is ever copied in.
  - On DGX/Pop: the same CoreWeave token, exchanged for temporary S3 credentials (`bundle.py s3_client`).
  - On DGX, `boto3 >= 1.42.5` comes from the client venv: `bundle.py` re-executes itself there, because the system
    boto3 is 1.34.46.

### Checkpoint sync back to DGX and Pop

- **Incremental publish.** While the job runs, `bundle.py watch` (started by the bootstrap) checks every
  `NATLANG_CW_PUBLISH_INTERVAL` seconds (default 60) whether a trigger file changed (`*.pt`, `report.json`,
  `*certificate*.json`). When one did, it publishes the whole output directory: objects, then the manifest, then
  `outputs/<job>/<dir>/latest.json`. So a reader always sees a complete state. Without a trigger change it still
  publishes every 30 minutes (`--max-interval`), which carries the logs.
  - Consistency: files ≤ 64 MiB are read into memory once, so a growing log is hashed and uploaded as the same bytes.
    A checkpoint is hashed and uploaded through one open descriptor, so the trainer's atomic rename during the upload
    cannot mix two versions.
  - Each checkpoint's 47 GB is hashed once per new inode (cached), and unchanged objects are not re-uploaded.
  - The pointer keeps the `previous` manifest key, so a resume can fall back one publish.
  - On exit the bootstrap stops the watcher, publishes once more, and writes `outputs/<job>/exit.json`.
- **Resume after expiry, stop or failure:** `cwctl.py launch JOB.json --resume`. It relaunches the same job id;
  the bootstrap (`NATLANG_CW_RESUME=1`) fetches each output directory's `latest.json` state, verified, at its
  original absolute path before running the job. The recipe then resumes from that `checkpoint.pt` as on DGX. On
  the first launch the pointer is missing, so `--missing-ok` makes this a no-op.
- DGX pulls with `bundle.py fetch --bucket B --pointer-key outputs/<job>/<dir>/latest.json`. That reads the
  pointer, fetches the manifest it names, and verifies every file. Add
  `--map /home/werg/natlang/runs/x=/home/werg/natlang/runs/x-cw` to land beside the local lineage instead of over it.
- Resume on CoreWeave: a new sandbox fetches the same run directory at its original absolute path, then runs the
  recipe with `--handoff` or the plain resume. The checkpoint identity is content-based, so moving a run between DGX
  and CoreWeave is a resume, not a restart (owner rule "continue training with more data": never restart).
- **Storage hygiene (AGENTS.md) applies to the bucket: `bundle.py gc`.**
  - Kept roots:
    - every job's `latest.json` pointer, with its latest and previous manifests, except jobs named with
      `--drop-job` (a discarded approach);
    - input bundles named with `--keep-bundle` or written within `--keep-input-days` (14).
  - Kept content: everything a kept manifest references (files and code).
  - Deleted: a dropped job's whole `outputs/<job>/` prefix, unkept manifests, and unreferenced `cas/` objects.
  - Nothing younger than `--min-age-hours` (24) goes, because an in-flight upload writes its objects before its
    manifest.
  - Without `--execute` it only prints the plan; `--out plan.json` writes the exact keys first (list, then delete).
    With `--execute` it deletes exactly those keys.
  - Run it at the storage-check cadence. Brief the other machine when a job is dropped.

### Which workloads move first

1. **Smoke** (CPU, then GPU). Check credentials, placement, image pull, egress, `sm_120` torch, and that a nonzero
   exit does not loop.
2. **`heldv2-base-mellum`** (`scripts/coreweave/jobs/heldv2-base-mellum.json`). This is the Mellum base task-success
   reference that gates the Mellum task bar and is otherwise stuck until the pause window. It is low-risk: read-only
   inputs, a 23 GB bundle, and results small and verifiable against the DGX plan's exact command and packet sha256.
3. **More task evals in parallel:** per-checkpoint task evals of v4 exports, LFM base references if Pop wants them,
   and the noise-measuring repeat runs the direct-stage rule needs ("noise measured by repeat runs"): N sandboxes at
   once.
4. **Mellum training on CoreWeave.** Start with an E2E arm or a v4 ablation, not the live v4 lineage. Measure
   throughput per update and the host-RAM peak against DGX. Then decide whether the main v4 lineage hands off to
   CoreWeave. That is a resume at a checkpoint boundary, a brief to Pop, and an owner-visible decision because it
   moves the active lineage.
5. **Parallel arms:** e2e dragnets, objective ablations and nested-family member sweeps, one sandbox each. This is
   where 10 sandboxes pay off. DGX keeps the main lineage or the teacher.

### Cost model

- Billing is on the request, per millisecond: CPU at $0.128 per core-hour and memory at $0.0212 per GiB-hour. The GPU
  rate is unknown (preview). The planning reference is $2.50 per GPU-hour, the on-demand RTX PRO 6000 per-GPU price.
- `cwctl.py plan --gpu-hour-price P` prints the bound for a job.

| Shape | CPU+mem/h | +GPU at $2.50 | Per day | 10 in parallel per day |
| - | - | - | - | - |
| Eval (16 CPU, 64 Gi, 1 GPU) | $3.40 | $5.90 | $142 (if run 24 h) | $1,416 |
| Mellum training (16 CPU, 128 Gi, 1 GPU) | $4.76 | $7.26 | $174 | $1,742 |
| Training, lean (8 CPU, 96 Gi, 1 GPU) | $3.06 | $5.56 | $133 | $1,335 |
| CPU render (16 CPU, 32 Gi) | $2.73 | – | $66 | $655 |

- Storage for 150 GB of bundles and checkpoints hot is about $9 per month.
- The main cost levers:
  - right-size memory, which costs more than CPU at these shapes;
  - stop sandboxes when the job ends: the bootstrap exits, so the sandbox completes and billing stops;
  - a lifetime only as long as needed, which also bounds a stuck job's cost.

### Risks

- **GPU capacity and quota for 10 parallel GPUs are unknown.** Placement errors are retried with backoff; there is no
  capacity guarantee.
- **The GPU price is unknown.** Get the quote before scaling past a few sandboxes.
- **Hard kill at lifetime expiry, with no grace.** Lifetimes are generous, outputs are published per checkpoint
  (follow-up), and sandboxes are stopped explicitly.
- **Egress policy is unknown.** If pip, npm and nodejs.org are blocked, switch to the prebuilt image (phase 2) with
  node included.
- **WIF setup needs org admin roles.** Until it is done, sandboxes cannot read buckets. The interim fallback is not
  to pass keys in env, which is forbidden by `job.py`, but a secret-store reference (`secrets=[…]`, `wandb` store),
  to be designed if needed.
- **Upload bandwidth from DGX for 60–130 GB per Mellum stage** is unmeasured. Content addressing means each byte goes
  up once.
- **Client and doc drift:** two mismatches were found already, and `cwctl.py plan` shows the client's real encoding.
- **Teacher-dependent jobs cannot move,** because :8082 is not reachable from CoreWeave.
- **Numerical parity:** different GPU and kernel timings. Compare the first CoreWeave eval against the DGX value for
  the same checkpoint before trusting it (owner rule: an unexplained gap is a bug).

## 4b. Local rehearsal (no keys)

`python3 scripts/coreweave/rehearse.py` runs the whole loop on DGX with a local S3 stand-in, so the first real
launch tests only CoreWeave itself:

1. Start `moto_server` from the client venv as a ledger unit.
2. Bundle `git archive HEAD` and a tiny input, and upload them.
3. `cwctl.local_docker_command` turns exactly the `Sandbox.run` request into a fresh `docker run`: same image,
   command, env and mounted bootstrap, nothing of the host mounted, and the DGX roots mirrored inside the container.
4. The job is `rehearsal_job.py`, a CPU stand-in that checkpoints by atomic rename. It runs while `bundle.py watch`
   publishes.
5. On the host: `exit.json` and the output pointer, then fetch and verify.
6. A relaunch with `--resume` and exit status 3 must continue from step 4 to step 8, record status 3, and not restart.
7. A dry gc plan.

Everything runs as low-priority ledger units, so on a full DGX it waits its turn.

## 5. Bring-up checklist (when the keys arrive)

Owner (once):

1. Create a CoreWeave API access token (console → Tokens) for a user with **Sandbox User** and **Object Storage
   Admin** (or an org policy with `cwobject:CreateAccessKey`). Confirm with the account team that the org is
   **GPU-sandbox allowlisted**, and ask for the GPU rate and the GPU concurrency limit.
2. Put the token in `~/.config/natlang/coreweave.env` on DGX, mode 600, never in the repo:
   `CWSANDBOX_API_KEY=…`. Add `NATLANG_CW_BUCKET=natlang-cw` and `NATLANG_CW_REGION=<bucket AZ, e.g. US-EAST-04A>`.
   `COREWEAVE_API_TOKEN` is accepted as an alias.
3. Create the bucket `natlang-cw` in an AZ with RTX PRO 6000 capacity (for example US-EAST-04A, which carries both).
4. Workload Identity Federation for sandboxes:
   - Console → IAM → Workload Federation: OIDC config with issuer **and** audience `https://oidc.cwsandbox.com`.
   - Org policies: `cwobject:CreateAccessKeyOIDC` on `*`, plus `s3:Get*`, `s3:List*`, `s3:Put*` and
     `s3:DeleteObject` on `natlang-cw` and `natlang-cw/*`, both for principal `role/https://oidc.cwsandbox.com:*`.
   - Then:

     ```
     curl -X PUT https://api.cwsandbox.com/v1beta2/object-storage/wif-config \
       -H "Authorization: Bearer $CWSANDBOX_API_KEY" -H 'Content-Type: application/json' \
       -d '{"wifConfigId":"<config id>","enabled":true,"allowedBuckets":["natlang-cw"],"maxPermission":"OBJECT_STORAGE_PERMISSION_READ_WRITE"}'
     ```

Agent (DGX), in order:

```sh
cd /home/werg/natlang
scripts/coreweave/setup_client.sh                       # client venv (already created 2026-10-12)
python3 scripts/coreweave/cwctl.py doctor               # credential: CWSANDBOX_API_KEY, bucket, region
python3 scripts/coreweave/cwctl.py smoke                # CPU sandbox, 15 min max; prints the sandbox id
python3 scripts/coreweave/cwctl.py logs <id> --follow   # uname, nproc, free, df /work
python3 scripts/coreweave/cwctl.py smoke --gpu          # RTX PRO 6000: torch sm_120, mem_get_info, a CUDA add
python3 scripts/coreweave/cwctl.py logs <id> --follow
python3 scripts/coreweave/bundle.py ping --bucket natlang-cw   # DGX-side object storage via token exchange
# first real job: Mellum base task reference
python3 scripts/coreweave/bundle.py plan --code-sha origin/main \
  --extra runs/neuralese-direct-mellum-20261011/heads-raw-token.pt --extra runs/natlang-held-task-packet-v2-20261012 \
  --extra /home/werg/data/models/mellum21-12b-a2.5b-thinking -o /tmp/cw/heldv2-base-mellum.json
python3 scripts/coreweave/bundle.py upload /tmp/cw/heldv2-base-mellum.json --bucket natlang-cw
python3 scripts/coreweave/cwctl.py plan scripts/coreweave/jobs/heldv2-base-mellum.json --bundle bundles/<id>.json
python3 scripts/coreweave/cwctl.py launch scripts/coreweave/jobs/heldv2-base-mellum.json --bundle bundles/<id>.json
python3 scripts/coreweave/cwctl.py logs <id> --follow
python3 scripts/coreweave/cwctl.py status-line          # paste into coord status --set
```

The model-directory hash reads 23 GB once. On the GB10 that page cache competes with CUDA, so run the `plan` step
under the memory ledger (priority low).

Check in the smokes:

- Does the sandbox complete (not restart) when the bootstrap exits 0?
- Is pip reachable? Is nodejs.org reachable?
- Does the WIF token reach the bucket?
- How long does the GPU sandbox take to start?

Record the answers here.
