# Memory admission on the DGX

The GB10 has one 121 GB pool. CUDA allocations come from it, systemd's `MemoryMax` does not see them, and the
kernel's OOM killer ranks processes by resident pages, which leave CUDA memory out. On 2026-10-04 an extra CUDA
server pushed the machine over and the global OOM killed the v10 teacher campaign along with it.

## Rules

- Launch every heavy job (anything with CUDA, a model load, or more than a few GB) through the ledger:

  ```sh
  python3 scripts/memory_ledger.py run --unit natlang-<job> --budget-gb 12 --class experiment -- <command>
  ```

  Admission requires `MemAvailable − outstanding claims − budget ≥ reserve` (default reserve 16 GB);
  `--wait SECONDS` queues instead of refusing. Outstanding demand of a running claim is its budget minus what
  it already uses (cgroup memory plus its processes' CUDA memory from nvidia-smi), so a job still loading
  counts at its full budget.
- The unit gets `OOMScoreAdjust` by class (experiment 900, collection 600, service 300). Unprivileged units can
  only raise their score; the campaign and the vLLM container stay at 0, so the kernel picks admitted jobs first.
- The job sees `NATLANG_CUDA_MEMORY_GB`; `label_decision_cases.py`, the Neuralese pilot and the Neuralese server
  use it as their default `torch.cuda.set_per_process_memory_fraction` cap. Load weights straight onto CUDA
  (`device_map`/`device=` at load) instead of building on CPU and copying: a copy holds both for a while.
- `natlang-memory-guard.service` (user unit, installed and enabled) checks every 5 s. It stops a unit over 115%
  of its budget, and when free memory drops below 8 GB it stops the lowest-priority, newest admitted unit. It
  never touches anything that was not admitted through the ledger.
- `python3 scripts/memory_ledger.py status` shows claims, use and headroom.

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
