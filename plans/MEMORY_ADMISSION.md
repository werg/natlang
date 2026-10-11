# Memory admission on the DGX

The GB10 has one 121 GB pool. CUDA allocations come from it, systemd's `MemoryMax` does not see them, and the
kernel's OOM killer ranks processes by resident pages, which leave CUDA memory out. On 2026-10-04 an extra CUDA
server pushed the machine over and the global OOM killed the v10 teacher campaign along with it.

## Rules (see "Oversubscription and priorities" below; its admission rule replaces the claim-sum rule)

- Launch every heavy job (anything with CUDA, a model load, or more than a few GB) through the ledger:

  ```sh
  python3 scripts/memory_ledger.py run --unit natlang-<job> --budget-gb 12 --class experiment -- <command>
  ```

  Admission requires `MemAvailable − outstanding claims − budget ≥ reserve` (default reserve 8 GB, the guard's
  floor); `--wait SECONDS` queues instead of refusing. Outstanding demand of a running claim is its budget minus
  what it already uses (cgroup memory plus its processes' CUDA memory from nvidia-smi), so a job still loading
  counts at its full budget. A claim that has run and been measured for 30 minutes counts only up to the highest
  use measured (the guard samples every 5 s): a server parked at 56 of its 62 GB no longer withholds the other 6.
  A later spike is what the floor is for, and the guard's first victim is the newest experiment. A job whose
  working set grows on its own schedule, or that checks MemFree before each step (the Maple warm-up's update
  preflight), is launched with `run --hold-budget` (or re-adopted with `adopt --hold-budget`, which keeps its
  admission time and measured peak): it always counts its whole unspent budget.
- The unit gets `OOMScoreAdjust` by class (experiment 900, collection 600, service 300). Unprivileged units can
  only raise their score; the campaign and the vLLM container stay at 0, so the kernel picks admitted jobs first.
- The job sees `NATLANG_CUDA_MEMORY_GB`; `label_decision_cases.py`, the Neuralese pilot and the Neuralese server
  use it as their default `torch.cuda.set_per_process_memory_fraction` cap. Load weights straight onto CUDA
  (`device_map`/`device=` at load) instead of building on CPU and copying: a copy holds both for a while.
- `natlang-memory-guard.service` (user unit, installed and enabled) checks every 5 s. It stops a unit over 115%
  of its budget, and when free memory drops below 8 GB it stops the lowest-priority, newest admitted unit. It
  never touches anything that was not admitted through the ledger.
- `python3 scripts/memory_ledger.py status` shows claims, use, measured peak, outstanding demand and headroom.
- Page cache: CUDA allocates only from MemFree, so when MemFree is short the ledger drops clean page cache of large
  files under the data roots (`release-cache`). One walk at a time (a lock file), at idle I/O priority, skipping
  hidden directories, and only when there is cache to drop (MemAvailable − MemFree ≥ 2 GB for the guard, 1 GB
  before an admission). A walk takes about two minutes; after one that freed under 1 GB the guard waits 30 minutes. On 2026-10-07 the guard started a walk every minute while the teacher and the warm-up kept
  MemFree below 16 GB with almost no cache to drop; 40 walks piled up on the external disk (90% busy), and any
  admission that needed a walk waited behind them.

## Campaign restarts

`scripts/systemd/natlang-campaign-resume@.service` runs a campaign's `root-launch.sh --resume` and restarts it
after an abnormal end (a signal such as an OOM kill), never after an error exit, which needs review. It waits
for the teacher server first and gives up after three restarts in three hours. Installing it for a campaign is
the owner's call:

```sh
cp scripts/systemd/natlang-campaign-resume@.service ~/.config/systemd/user/ && systemctl --user daemon-reload
systemctl --user start "natlang-campaign-resume@$(systemd-escape --path campaign-qwen36-current-train-refresh-20261003-v10/pool-v10-single-source-v2).service"
```

Run under the unit, `KillMode=control-group` guarantees that no process of the previous run remains, which is
what `--resume` requires.


## Stop grace scales with the job (owner 2026-10-10)

Owner: "we want to snapshot the optimizer state when the signal arrives — can't we just give it more time", and "the
guard should be different for smaller models". Every stop now gives the job time to write its full state:

- **Grace per unit.** The shared checkpoint writer (`train/trajectory_state.atomic_checkpoint`) logs every completed
  write (bytes, seconds, `NATLANG_LEDGER_UNIT`) to `~/.local/state/natlang/checkpoint-writes.jsonl`; `run` passes the
  unit name into its job (containers: `-e NATLANG_LEDGER_UNIT=<unit>`). A unit's grace is its family's largest recent
  checkpoint over the median measured write throughput, times 2, plus 15 s to reach a step boundary (at least 5 s; a
  family that never checkpointed, such as tests and probes, gets 5 s). `run --stop-seconds N` overrides it. Mellum
  (46.8 GB at ~190 MB/s) gets ~8 min, a 350M LFM warm-up a few seconds.
- **Planned stops** (`systemctl --user stop`): `TimeoutStopSec` = grace + 30 s and `ExecStopPost docker stop -t
  grace`; launchers also create containers with `--stop-timeout` (runs/mellum-foundation-qat-20261010/launch.sh: 600).
- **Guard stops.** Page cache release runs first, as before. Then one victim at a time gets SIGTERM (the unit and,
  for containers, `docker kill -s TERM` to the job inside) and its grace while the guard watches; no second victim is
  picked meanwhile. It is killed when the grace runs out, or at once when available memory falls below the emergency
  level (3 GB, `guard --emergency-gb`, below the 8 GB floor), the same for every size.
- **Victim order.** Over-budget units under pressure first (unchanged). Below the floor: class priority first
  (unchanged), then within a class the unit that frees the most memory per second of grace (small, stateless jobs
  before big long-running trainers), then the newest.
- **No host copies on the stop path.** Checkpoints are written from device tensors (text warm-up synchronous
  saves, `backbone_trainable_state`, qat_convert), so a stop does not double the model in unified memory.


## Oversubscription and priorities (owner 2026-10-11)

Claims are admitted against real headroom, not the sum of declared budgets (87 GB claimed while 5 GB was used blocked a
priority GPU job, which then got a reduced cap and OOMed). The sum of claims may exceed the machine.

- Reservation of a running claim: its unspent budget in full while ramping up (5 min) or with `--fixed-claim` (alias
  `--hold-budget`); half for a GPU unit (auto-detected from nvidia-smi pids of its cgroup/container); a quarter for a
  CPU unit. Observed use is cgroup memory (cache and MADV_FREE excluded) plus CUDA per pid plus the attached container.
- `--priority high|medium|low` (default medium, services high). High: training, GPU end-to-end/direct runs. Medium:
  smokes, evals. Low: data generation, rendering, check-main, other CPU jobs. Callers pass it; the class defaults do not
  guess. Priority orders the waiting queue (head only is admitted), sets the kernel OOM score (200/600/900), orders the
  guard's victims under pressure (lowest first; a unit over its own budget goes first), and picks preemption victims.
- Preemption: the guard stops (graceful, with the usual stop grace) the lowest-priority resumable units for the queue
  head when real headroom is short and stopping all candidates would make it fit. `--resumable` (default yes except
  services) means the command resumes from its checkpoints; such units are requeued (spec stored in the claim) and
  relaunched by the guard after 2 minutes when they fit. Over-budget stops are not requeued.
- Kernel safety net evaluated 2026-10-11: user units can set OOMScoreAdjust but the floor is the user manager's own
  score (100): -900 is silently clamped. systemd-oomd is not installed. Slices with MemoryLow need delegation
  (`Delegate=no` for the user manager) and do not see CUDA. Decision: use positive OOM scores by priority as the instant
  safety net (kernel kills the +900 jobs first), keep the guard for graceful, checkpointed stops, requeue and preemption.
  The guard floor is 10 GB and polls every 2 s (was 8 GB / 5 s) because CUDA allocation bursts can outrun it.
- `status` shows priority, budget, use, reserved, resumable, and the waiting queue with `would_preempt`.
- Superseded: the claim-shrinking design (effective claims, commit 4f32386a) was dropped.

- Follow-up 2026-10-12: (1) detached containers: on preempt/pressure stops the guard finds the unit's containers (command, `run --container`, `docker run --name` client in the cgroup, unit name), signals them, `docker stop`s survivors when the unit exits, and does not relaunch a requeued unit while one runs (a wrapper shell runs its TERM trap only after its foreground `docker run` ends, so the unit's end never stopped the container). (2) Undeclared units are resumable (tests, checks, data jobs; only services and adopted units are protected; `--no-resumable` for long non-checkpointed jobs); `status` waiting entries list `blocked_by` (non-preemptible units with reason) and refusals name them.
