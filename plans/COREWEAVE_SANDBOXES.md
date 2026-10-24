# CoreWeave serverless GPU sandboxes

Status 2026-10-12 (DGX). The owner has preview access, and the keys are not set yet. Everything below that needs no
key is built: the controller, the bootstrap, content-addressed bundles and tests (`scripts/coreweave/`). The first
commands to run once the key arrives are in "Bring-up checklist". The goal is to take training and eval pressure off
the DGX GB10, whose 121 GB unified memory is the bottleneck. Today the Mellum v4 run holds 98 GB of it, and the Mellum
base task reference (34 GB) waits for a pause window.

What runs inside a sandbox and how fast (hardware facts for the RTX PRO 6000 / sm_120, the training and teacher
serving plans, hardware profiles, the image and the first-sandbox benchmark jobs) is in
[COREWEAVE_HARDWARE_OPTIMIZATION.md](COREWEAVE_HARDWARE_OPTIMIZATION.md).

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

Everything runs as medium-priority ledger units (the smoke class), so on a full DGX it waits its turn.

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

## 6. Auth we have, and the data path without a bucket (2026-10-12)

The owner authorized the W&B key for sandboxes. It lives in `~/.config/natlang/coreweave.env` (mode 600) with
`CWSANDBOX_AUTH=wandb`, `WANDB_ENTITY=werg` and `WANDB_PROJECT=natlang`. `cwctl.py` picks the strategy from
`CWSANDBOX_AUTH` (or a leading `--auth wandb|coreweave_api_key`). It never prints the key or forwards it into a
sandbox.

Pop has the same private env file and client setup as DGX (2026-10-14). Run
`scripts/coreweave/setup_client.sh` on either machine; it installs the W&B auth
extra needed by `cwctl.py ls/get/logs`. The machine that launches a sandbox
still owns its lifecycle and launch record; the other machine can inspect it
without altering the job.

What W&B auth does not give us:
- **Runner discovery.** `list_runners` answers "W&B auth is not allowed for this sandbox operation".
- **The object-storage identity.** `object_storage_access` needs a WIF configuration registered by a CoreWeave
  `SANDBOX_ADMIN` token for an org with an AI Object Storage bucket. A W&B-billed org has neither. Untested: the
  sandbox create would be refused, and we create sandboxes only for real jobs.
- **The token exchange.** It needs a CoreWeave API token.

So, until the owner creates a CoreWeave org token, there is no bucket. The owner would create: a CoreWeave API
access token with Sandbox User, Sandbox Admin and Object Storage Admin, a bucket, and the WIF setup in section 5.
Data takes one of these paths, all keeping the bundle manifest as the contract and verifying sha256 on arrival:

1. **Public URLs, verified.** A bundle entry may carry `url`.
   - Mellum2.1 is public and ungated on Hugging Face. The DGX copy `/home/werg/data/models/mellum21-12b-a2.5b-thinking`
     is the HF download at revision `92ddae9f`: its `.cache/huggingface/download/*.metadata` etags equal the HF LFS
     sha256s (`model-00001…` = `f5ccf801…`).
   - So the sandbox downloads the 23 GB from `https://huggingface.co/<repo>/resolve/<rev>/<file>`, at CoreWeave's
     bandwidth, not DGX's uplink, and checks each file against the manifest's sha256.
   - The same works for public base models, the LFM QAD source, and public datasets.
2. **The sandbox file API, controller-pushed.** This is for our own small and medium files: the code snapshot (58 MB),
   heads (124 MB), packets and union shards (GB).
   - `write_file_streaming` into a CAS inbox in the sandbox. The unary file calls are capped at 32 MiB; streaming is
     the large-file path.
   - The bootstrap waits for an `inputs-ready` marker, then `bundle.py fetch` moves inbox objects into place after
     verifying them.
   - Outputs come back with `read_file_streaming`, verified against the manifest the bootstrap writes.
   - A completed sandbox has no file API, so in this mode the bootstrap waits for an `outputs-collected` marker (or a
     timeout) before it exits.
   - For training this is `cwctl.py collect --follow`, which pulls each new checkpoint. That is the same
     incremental-publish contract, with the controller pulling.
3. **W&B Artifacts as the transport.** This needs a W&B credential inside the sandbox. The only acceptable way is
   the platform's secret store, `secrets=[Secret(store="wandb", name=<W&B secret>, env_var="WANDB_API_KEY")]`,
   which the owner creates in W&B. We never copy the key in.
   - Strengths:
     - content deduplication and versioning, with aliases such as `latest`;
     - lineage (run → artifact);
     - one auth for DGX, Pop and sandboxes;
     - good fit for checkpoints and run outputs.
   - Weaknesses:
     - the digests are W&B's, not sha256 (we ship our manifest inside the artifact and verify our hashes on
       arrival, as now);
     - W&B storage pricing and per-file limits for 47 GB checkpoints are unknown;
     - throughput from CoreWeave to W&B storage is unmeasured.
   - Verdict: a good **output and checkpoint** path once the owner creates the W&B secret. Inputs are better served by
     paths 1 and 2, since public weights need no storage at all.

Order of use: URL inputs plus API push now. Then W&B Artifacts for outputs when the owner adds the secret, or the
CoreWeave bucket when the owner adds an org token. Either replaces path 2's controller pull for long training.

## 7. Migration plan: move training and teacher generation off DGX (owner target state, 2026-10-12)

Target state:
- **Sandboxes:** Mellum v4 and its successor stages, the e2e stage, ablation and dragnet arms, task evals and base
  references, teacher data generation (serving plus collection lanes).
- **DGX:** investigations, tests and prototyping.
- **Working ceiling:** about 20 concurrent GPUs. It is learned passively: an occupancy-quota error on a normal create
  is recorded as `quota_observed`. No probing.
- **Creates:** throttled. They are serialized by a lock, at least 45 s apart (`NATLANG_CW_CREATE_INTERVAL`), with
  exponential backoff and jitter on 429, `CWSANDBOX_SERVERLESS_CREATE_THROTTLED` and `RESOURCE_EXHAUSTED`, and
  never a tight loop. Long-lived sandboxes are preferred, with several jobs exec'ed into one.
- **Training lifetime:** training sandboxes get the maximum lifetime, 30 days (`job.py` default for `kind: train`).
  The bootstrap exits when the job ends, so the sandbox completes then, and the lifetime is only a ceiling.
- **Cost:** the preview is free; the only limit is the GPU quota.

Steps, in order. Each step needs the previous one.

**(a) GPU smoke.**
- Needs: the public `nvcr.io/nvidia/pytorch:26.09-py3` (x86_64), the bootstrap, and nothing else.
- Answers:
  - GPU model, memory, driver and CUDA from `nvidia-smi`;
  - torch `sm_120`;
  - egress to pypi, nodejs.org, npm, Hugging Face, GitHub and cwobject.com;
  - start-up time;
  - whether exit 0 completes the sandbox.
- Results: section 8.

**(b) Mellum base held-task reference** (`jobs/heldv2-base-mellum.json`).
- Data:
  - the model by verified HF URLs;
  - the code snapshot, heads and packet pushed through the file API.
- Needs node ≥ 22 and a ts-host build in the sandbox. The bootstrap installs them, checked against `SHASUMS256.txt`.
- Output back: `runs/heldv2-base-mellum-full/` pulled and verified.
- Check: compare with the DGX value when DGX's pause-window run exists.

**(c) Teacher serving plus one collection lane, end to end.**
- Engine: vLLM with the DGX teacher's exact flags (`nvidia/Qwen3.6-35B-A3B-NVFP4`, `--max-model-len 65536`,
  `--kv-cache-dtype fp8`, `--enable-prefix-caching`, the qwen3 tool and reasoning parsers). `--gpu-memory-utilization`
  rises from 0.45 to about 0.90, because nothing else shares the 96 GB. NVFP4 is native on Blackwell `sm_120`.
- Image: the public `vllm/vllm-openai` amd64 image. The serving-engine choice and tuning belong to the hardware agent
  (plans/COREWEAVE_HARDWARE_OPTIMIZATION.md).
- Topology A (first): **teacher and lane in one sandbox.**
  - vLLM serves on `127.0.0.1:8082`, the same URL the lanes use on DGX. The node collectors (ts-host) run beside it.
  - No public port, no auth surface, one create.
  - The collection outputs (`*.ir.jsonl`, turns, SFT rows) are published as a bundle manifest. DGX registers them in
    `training/neuralese_corpora.json` with an immutable corpus manifest under `training/corpus-manifests/`, as for
    any lane. Copying is not admission.
- Topology B (later, for DGX- or Pop-side lanes): **a long-lived serving sandbox with a public HTTPS endpoint.**
  - Sandboxes can expose ports: `services=[Service(port, visibility=PUBLIC, endpoint=Endpoint(kind=HTTPS,
    auth=OPEN))]`. The URL appears in `service_urls` once the sandbox is RUNNING. At most about 5 services per
    sandbox.
  - The platform does not authenticate callers, so vLLM runs with `--api-key` set to a per-sandbox token. `cwctl`
    generates it, keeps it in `~/.local/state/natlang/coreweave/tokens/<job>` (mode 600), and passes it as a
    declared endpoint token (not an owner credential).
  - Other sandboxes can reach it the same way, or through `EgressRule(tenant=…)`.
- Needs:
  - the lane's inputs (selection/case IR JSONL, small) by API push;
  - the model by HF URL (public);
  - throughput measured against DGX :8082 on the same lane.

**(d) One e2e or ablation training arm.**
- Data: the export or checkpoint it starts from (via the file API until a bucket or W&B Artifacts exists), union
  subsets, heads.
- Lifetime 30 days. `bundle.py watch`-style incremental checkpoint pulls (`cwctl.py collect --follow`).
- Checkpoint path back: pulled into `runs/<arm>-cw/` by `--map`, verified.
- Throughput and host-RAM peak are compared with DGX. Batch size, compile and FP8 settings come from the hardware
  agent's benchmark job.

**(e) Mellum lineage handoff to a sandbox.**
1. The v4 run directory (`checkpoint.pt` 47 GB plus state) goes up at the same absolute paths, through the mirror
   roots.
2. The sandbox runs `recipe.py --handoff`, which resumes from that exact checkpoint and optimizer state.
3. Incremental checkpoint publishing brings every checkpoint back to DGX.

Policy (`placeJob.nl` step 2): moving the active lineage is the owner's decision, so this step needs the owner's go,
and the coordinator briefs Pop.

### What leaves DGX first (proposal, from the ledger on 2026-10-12 20:40Z)

The ledger at that moment:
- **Running:** Mellum v4 (102 GB claim, 66 GB used), the s7 RL GRPO round, an LFM base dragnet with its eval, and one
  e2e dragnet arm.
- **Waiting (17 units):**
  - three more e2e dragnet arms (12 GB each);
  - e2e tests and perf-e2e (10–13 GB);
  - the harness matrix, the specializer dry run (18 GB), s7 RL round 1, memetic author, the hardware agent's tests;
  - data mining and assembly (low priority).
- **Effect:** a 1 GB smoke waited more than an hour.

Order, so that DGX memory pressure goes away:

| Order | Workload | Why it goes early | Needs |
| - | - | - | - |
| 1 | e2e dragnet arms (frozen, adamw, head, muon…) | independent arms of 12 GB each: one sandbox per arm, many in parallel | the export and checkpoint they start from, union subsets, heads (API push); outputs collected |
| 2 | task-success evals and base references (heldv2 Mellum, per-checkpoint evals, noise repeats) | read-only and verifiable against DGX numbers | model by HF URL, ts-host build in the sandbox |
| 3 | s7 RL rounds (GRPO), memetic author and screens | small to medium GPU, long queue waits | the round's inputs; a teacher only where the round calls one (then step c's sandbox) |
| 4 | perf-e2e, the harness matrix, the specializer dry run | batchable benchmark-style jobs | code plus small inputs |
| 5 | teacher serving plus collection lanes (step c) | frees the 59–74 GB vLLM teacher on DGX | vLLM image, HF model, lane inputs |
| 6 | Mellum v4 itself (step e) | the largest claim (102 GB); last, because it moves the active lineage | owner go, checkpoint upload (47 GB), incremental collect |

What stays on DGX: tests (check-main), investigations, prototyping, data assembly that reads the local corpora, and
the LFM lines' Pop-side work.

## 8. Measured on CoreWeave (W&B auth, entity `werg`, project `natlang`; 2026-10-12)

**Works:**
- `Sandbox.list` and `cwsandbox ls`.
- `snapshots list` (empty).
- A GPU create is **accepted**, with no `CWSANDBOX_GPU_NOT_ALLOWED`, so the allowlist is not refused at create.

**GPU sandboxes never start.** Two GPU smokes, created one at a time and 61 min apart:

| Sandbox | Image | Timeline | Outcome |
| - | - | - | - |
| `8cdea0e6-bafa-4b7f-aca1-ba850c57c28a` | `nvcr.io/nvidia/pytorch:26.09-py3` | `creating` for 61 min | stopped by us; it then showed runner `sandbox-prod-us-east-15a`, container still `creating`, never started |
| `2c207a7e-d710-4c5f-b0d2-b5178a0ec202` | `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` (small) | `creating` for 33 min; the control plane then answered `UNAVAILABLE: the sandbox is being reconciled with its runner; retry shortly` | `failed` on runner `sandbox-prod-us-east-15a`; container never left `creating`; no return code |

- The runtime class was `cw-kata-nvidia-gpu`.
- US-EAST-15A is where the RTX PRO 6000 Standard Memory nodes are.

**`project not found`.** With W&B auth, every call that touches the tenant's CoreWeave-side resources fails with
"project not found":
- reading the failed sandbox's logs (`Connect to sandbox data plane failed: project not found`);
- `get_snapshot_bucket_config`;
- `Volume.list`.

It happens with or without `WANDB_PROJECT`/`WANDB_ENTITY`, and the W&B project `natlang` exists in entity `werg`.
So the W&B-billed tenant appears not to be provisioned on the CoreWeave side (no backing project for the data plane,
snapshots or volumes). That probably also explains why the sandboxes never started. This is for CoreWeave to fix:
**the owner should send forge-support@coreweave.com the two sandbox ids, the runner, and the "project not found"
error.**

**Discovery:**
- `list_runners`: "W&B auth is not allowed for this sandbox operation".
- `get_runner` requires a CoreWeave `organization_id`.

So runner discovery needs a CoreWeave org token. Nothing in the W&B path exposes it.

**Default egress policy** (`effective_egress` of the sandbox): all public IPv4 (`0.0.0.0/0` minus the private,
link-local and multicast ranges), so pip, npm, nodejs.org, Hugging Face and GitHub should be reachable. The
in-sandbox probe has not run yet.

**Open until a GPU sandbox runs:**
- GPU model, memory, driver and CUDA (`nvidia-smi`);
- whether exit 0 completes the sandbox;
- measured egress;
- start-up time.
The smoke records all of these: `cwctl.py smoke --gpu`, then `--attach` for an existing one.

**Storage and discovery within the W&B org, the options:**
1. **File System Snapshots.** CoreWeave manages the bucket by default, so no bucket of ours is needed. This is the
   W&B-org way to persist and fork a sandbox's directory, e.g. an "inputs" snapshot restored into many sandboxes, or
   checkpoint suspend and resume. Blocked today by the same "project not found"; retest once support fixes the
   tenant.
2. **The sandbox file API** (`write_file_streaming` / `read_file_streaming`). Our `transport: api`. It needs the
   data plane, so it is blocked by the same error today.
3. **Public URLs** (HF), verified by sha256. Independent of the tenant.
4. **W&B Artifacts.** DGX and Pop use the W&B key directly. Inside a sandbox the platform injects a W&B secret
   (`Secret(store="wandb", name=…)`); the owner creates that secret in W&B, ideally a service-account key scoped to the
   team, and we never copy a key. This is the W&B-native storage for outputs and checkpoints, and it works across all
   machines.


## 9. Vendor-fix retest, 2026-10-14

The W&B tenant and default data-plane blockers are fixed in measured tests with
cwsandbox 1.19.0. `auto` and explicit `gateway` exec/file roundtrips now work;
`get_snapshot_bucket_config` returns `CW_MANAGED`. Mounted files also work.

GPU sandbox `a60ee9b5-258d-4d97-8ef7-fbda04242979` ran the same rich request
as `3d8a0c75-d951-478c-a90a-bb9f9984657d`, omitting only the scratch volume:
PyTorch `2.8.0+cu128`, CUDA 12.8, four CPUs, 16 GiB RAM, mounted file, env,
tags and one-hour lifetime. It reached running in 93 seconds. On RTX PRO 6000
sm_120, a deterministic tiny network completed 30 AdamW backward/optimizer
updates, loss 1.078510 -> 0.053739; saved and reloaded model/optimizer state,
then produced bitwise-identical finite CUDA inference. This qualifies the basic
GPU mechanism; it does not qualify natlang model throughput, kernels, task
quality, serving endpoints, or full training resume.

Plain control `df5c4154-01c4-4322-a527-e21b2bccd579` booted in 23 seconds;
mounted-file control `3a83865e-fdfe-4e71-8926-441360f76c8c` in 52 seconds.
Both passed default/gateway exec and file roundtrips. Scratch-only control
`7c62f59b-8ba3-4719-aa58-489fa7123d31` remained creating through a four-minute
observation window and was explicitly stopped. Rich requests with scratch
volumes, including default-image `941d1bb3-fb75-4be2-8b6a-079e83ff4e18`,
remained creating for about 11 minutes and was explicitly stopped. The rich
PyTorch volume request remained creating for 15 minutes, then was explicitly
stopped. Neither produced a vendor terminal failure or useful state reason.
These controls isolate scratch-volume provisioning as the remaining suspected
blocker, without treating an observation timeout as a platform terminal error.

Use an explicit job declaration `"scratch": {"path": "/work", "size":
"50Gi", "storage": "root"}` to omit the vendor volume and retain the existing
bootstrap/input/output paths. The work directory lives on ephemeral root;
size is a planning estimate rather than reserved capacity. Collect verified
outputs before completion/stop. Existing declarations still request volumes.
No live lineage moved. The successful CUDA diagnostic was explicitly stopped.

Reproducible bounded diagnostics and JSON receipts are in
`/home/werg/data/dgxops/coreweave-{retest,control}-20261014.py` and
`coreweave-{pytorch-root,controls}-20261014.json`; their local units run through
the memory ledger. HF, PyPI, Node and GitHub HTTPS egress each returned 200.
No credentials were placed in a sandbox or output record.


The existing `cwctl.launch` -> bundle streaming -> bootstrap -> `cwctl.collect`
path also passed using root storage in sandbox
`46fad1e7-7cea-4c74-9b67-ded7ca43e829` (running after 19.9 seconds).
The code snapshot was 60,948,359 bytes. The job repeated the 30 CUDA updates,
then exported checkpoint and result JSON to DGX with per-file SHA-256 verification:
130,671 bytes, output bundle
`4d3c5c50dd8f8d88e0f3af4dd2d0f8b8213adfcd4fe18a731acd7f581ee93313`.
After sandbox stop, a separate ledgered DGX check loaded the model and optimizer,
verified all optimizer states at step 30, and ran finite CPU inference.
Local artifact directory: `/home/werg/data/dgxops/coreweave-native-loop-output-20261014`.
Repeat the exact loop with
`memory_ledger.py run --unit <unique-unit> --budget-gb 2 --priority medium --no-resumable -- /home/werg/.local/share/natlang/cw-venv/bin/python -u /home/werg/data/dgxops/coreweave-native-loop-20261014.py`.

All seven test sandbox IDs in this section were independently re-read after
cleanup and were terminal `completed` (user-requested stop). Exact cleanup
receipt: `/home/werg/data/dgxops/coreweave-cleanup-audit-20261014.json`.
No allocated diagnostic sandbox remains. The first malformed local mounted-file
request failed SDK validation before creating a sandbox; it used the documented
list keys rather than the installed SDK's mapping form, and was corrected locally.

Validation: 45 CoreWeave tests passed, two optional SDK tests skipped in the
local training environment; the actual SDK launch/stream/collect loop passed
remotely. Ledgered quick-main checks passed. The full-main run for commit
`65e8a69135` was preempted during the test suite by training admission and remains
pending; its partial output is not a full-main pass. NGC CUDA 13.4 kernels,
large-model training/resume, production teacher serving and actual filesystem
snapshot restore still need their declared benchmark/qualification jobs.


## Recovery incident and launcher update (2026-10-23)

At 19:52:35 UTC the two learners, self-hosted teacher and two evaluators
became Completed. Their observed specifications had no volumes or volume
mounts. Public SDK and authenticated gateway ReadFile requests reject them
as not running. Direct snapshot requests for both learners and the teacher
returned `this sandbox has no file system snapshot storage`. Backend container
Running fields do not prove the underlying files are still accessible. Provider
retention/recovery remains unverified.

Private exports survive independently: LFM's step-35690 full model/optimizer/RNG
checkpoint has been downloaded and SHA-256 verified locally; Mellum's exported
step-35223 full state is being fetched. Mellum's newer step-36223 state was
saved on its sandbox but has no located durable export. All 197 exported
internal teacher archives were downloaded and hash-verified. Actual range GETs
and exact-length checks also passed for 523 archived input/runtime bundles;
those checks establish current readability, not a new full-object SHA check.
Current custody and provider recovery details live under
`/home/werg/data/dgxops/full-service-recovery-root-20261023-v1`.

The shared job mechanism now accepts `resources: {"gpu": 1}` and an optional
explicit RAM requirement without inventing CPU/RAM defaults. Cost estimates
remain unknown where assigned CPU/RAM are unknown. A volume declaration may
include `scratch.restore_from_snapshot_id`; this is serialized through the
SDK's named filesystem-snapshot mount, preserving the requested mount and size.
Root storage cannot accept a restore selector. A filesystem restore does not
automatically qualify or resume a model; its exact checkpoint, source, optimizer,
RNG and exposure still need the normal continuation checks.

Offline request serialization and launcher tests do not establish working
backend snapshots. New production allocations need explicit recovery storage
and a periodic snapshot operator in addition to full-state/source exports.
The first READY snapshot and a verified restore must be observed before
snapshot recovery is reported as covered. Keep snapshot failures visible and
retain the latest independently verified backups while resolving them.

### Quota-independent recovery preparation, October 24

Owner correction: filesystem snapshots may require a new running sandbox to
access their contents, precisely when an exhausted account cannot allocate one.
Our successful salvage used private W&B run-file exports fetched directly on
DGX with no running sandbox. Make those independent full-state and source/control
exports the primary recovery route; snapshots supplement them. Keep optimizer,
RNG, per-row exposure, full evolving recipes, exact reader/runtime/control
closures, source/held inputs and generation journals. A model-only export does
not suffice. Keep one verified local recovery representation and bounded private
parents, with actual remote-presence checks before pruning.

The current preparation mechanisms are shared across both model lines:

- `scripts/coreweave/recovery_inputs.py --catalog CATALOG --out DIRECTORY`
  authenticates the original full private-export receipts, downloads exact
  content-addressed objects, independently hashes complete bytes and inventories
  TAR members without extraction. Original JSON custody receipts and gzip runtime
  archives are supported alongside zstd source archives. The default free-space
  floor is 60 GiB; failures retain the durable source and cannot replace a
  mismatching destination. Credentials and signed URLs are never journaled.
- `scripts/coreweave/recovery_readiness.py --metadata CHECKPOINT_METADATA
  --candidates SHA_TO_LOCAL_PATHS --archive-objects DIRECTORY --out REPORT`
  reconciles checkpoint-declared source trees and exact control/base bindings.
  It separates verified local files, actual verified archive members and missing
  dependencies. This is an input audit, not complete controller closure,
  admission, device qualification or resumed-training credit.
- `scripts/coreweave/recovery_snapshots.py --sandbox-id ID --state-dir DIRECTORY`
  schedules snapshots with durable request IDs, polling CREATING operations
  instead of generating duplicates after ambiguous RPCs. READY and successful
  restoration remain separate. `--verify-restore SNAPSHOT_ID
  --destination-sandbox-id DESTINATION --restore-files EXACT_FILES_JSON` checks
  the destination's actual backend restore-volume selector and hashes the real
  files there before writing a restore receipt. It does not create that sandbox.
  Retention touches only snapshots in this operator's own state, writes the exact
  deletion plan first, preserves the newest two plus the verified/protected
  parents, and requires a newer successful restore before deleting older states.
  No successful cloud capture or restore is claimed by local tests.

The owner reports at most $500 additional access, possibly less; actual remaining
credit and GPU rates are unknown. Do not reuse the old preview-free assumption
as a spending estimate. The lean prepared placement is local GB10 LFM, one cloud
Mellum learner and burst Qwen3.6 generation when existing verified supply needs
expansion. Measure useful throughput before adding a second cloud learner or a
dedicated evaluator. Bound spending using actual account rates and allocations;
snapshot access never counts as independent backup access at quota exhaustion.
