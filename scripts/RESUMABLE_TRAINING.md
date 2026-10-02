# Resumable training supervisor

`resumable_training_supervisor.py` runs one immutable training command at a
time. The plan is the run identity; edits to its command, pins, or output path
are rejected after the first start. It holds an exclusive `flock` while a
trainer child is active and records its state in the plan's `status_file`.

The plan schema is `natlang-resumable-training/1` and requires:

- `run_dir`, `checkpoint_state`, `status_file`, `lock_file`,
  `operator_stop_file`, and `stdout_log` absolute paths.
- `argv` as the complete command array, `cwd`, and explicit environment
  overrides in `env`. Docker mounts, data/output paths, and image reference
  belong in `argv`.
- `artifact_sha256`: absolute host paths mapped to SHA-256 digests. When using
  Docker, also set `runtime` to `{ "kind": "docker", "image_ref": "...",
  "image_id": "sha256:..." }`; both daemon readiness and exact image ID are
  checked before launch.
- `checkpoint_identity`: nonempty fields that must equal the same keys in the
  trainer checkpoint's `corpus` object (normally `data_sha256`, model revision,
  optimizer/settings identity, and target examples).
- `completion`: a pinned `step` and/or `trained_examples`. Set
  `final_evaluation_required: true` unless the exact command deliberately has
  `--skip-heldout-loss`; complete status then requires a numeric
  `heldout_after` in the final checkpoint.
- `retry`: bounded `max_attempts`, `initial_delay_seconds`, and
  `max_delay_seconds` for exponential backoff.

Example control commands:

```sh
python scripts/resumable_training_supervisor.py run PLAN.json
python scripts/resumable_training_supervisor.py status PLAN.json
python scripts/resumable_training_supervisor.py stop PLAN.json
python scripts/resumable_training_supervisor.py resume PLAN.json
```

`stop` writes a persistent operator latch and signals only the supervisor PID
whose Linux start time still matches the status receipt. `resume` clears that
latch; then start the service. A direct SIGTERM (including `systemctl --user
stop`) is a graceful pause without the latch. The trainer finishes its current
optimizer step, checkpoints, and exits; the next service start resumes the same
command and state. The supplied user service template is
`scripts/systemd/natlang-training-resume.service`. Install it under
`~/.config/systemd/user/`, set the user's service to enabled, and retain
`loginctl` linger so it starts after reboot. The template does not install or
start itself.

Checkpoint publication fsyncs every file and directory before the atomic
directory swap. Normal SIGTERM pauses at an optimizer-step boundary. A sudden
power loss or SIGKILL can still lose work since the previous periodic save;
recovery restores the last complete checkpoint, including optimizer,
scheduler, and RNG state.

The `status PLAN` command combines the saved lifecycle status with a fresh,
validated checkpoint observation and reports whether the recorded supervisor
process is still alive. It does not rewrite the lifecycle file. During a long
run, use this command or the checkpoint itself for progress; the saved status
file is updated only at lifecycle transitions.
