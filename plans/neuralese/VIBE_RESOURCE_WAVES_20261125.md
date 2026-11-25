# Keep deployment limits local to deployment work

Observed in actual V34 wave 0065: one of 64 cases requires an installed
deployment, so `collection_limits` reduces all 64 cases from four workers to two.
The remaining 63 ordinary cases do not need this resource limit.

Mechanism inventory before editing:

- `scripts/neuralese_data/vibestudio_campaign.py:collection_limits` applies the
  existing nested deployment cap. Keep its safety behavior for every selected
  wave, including the explicit legacy mixed-wave option.
- `next_wave` groups by workspace and review requirements. Add deployment
  requirements to this grouping, and rank groups by their usable worker slots.
- `drive` is the only production caller of `next_wave`; pass both the existing
  deployment worker limit and the explicit mixed-wave opt-out.
- `prerequisite_holds` uses the same case metadata to hold unavailable installed
  packages. Preserve those holds and every goal's ID, split and exposure history.
- Host `scripts/collect-live-workflow-teacher.mjs:groupLiveTeacherCases` batches
  the selected cases by workspace/baseline/devices and runtime lifetime. It uses
  the driver's effective concurrency; no Host or runtime protocol change is
  needed when the selected wave has one resource class.
- Native IR/export/ingestion and model backends consume execution receipts, not
  this scheduling grouping. No data format or model policy changes apply.

Default: resource-homogeneous waves. `--mixed-deployment-waves` restores the
previous grouping; the existing cap still prevents unsafe nested concurrency.
Fresh teacher coverage, family rotation, review homogeneity, workspace fairness,
tail filling and all 1,748 training goals remain.

Validation: targeted scheduler tests must cover ordinary throughput, deployment
limits, fresh deployment work ahead of ordinary retries, worker-sized tails,
eventual deployment selection and the explicit opt-out. No paid probe is needed.
The existing useful collection continues; the ready driver enters at a safe
boundary.
