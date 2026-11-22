# Release-aware Vibestudio attempt accounting

The retry limit belongs to the release that attempted the task. Two failures under an older harness do not establish exhaustion under a repaired harness. This is exact campaign accounting, with no change to model instructions, outcome policy, held separation or teacher selection.

Mechanism inventory:

| Component | Responsibility |
| --- | --- |
| `campaign_scope` | Current held-separated task identities, unchanged. |
| `attempt_context` / release validation | Snapshot the validated release SHA, pin bytes, catalog digest and driver/Host code provenance once per controller. Default epoch is that release SHA; `--attempt-epoch` explicitly overrides it. |
| Wave jobs | New plans record their epoch and immutable provenance. Historical plans without an epoch remain historical. |
| Execution records | Record the actually admitted subset separately; never rewrite the original plan. |
| `receipt_history` / `publish_status` | Associate unchanged receipts with their original plan epoch. Current-epoch teacher attempts determine exhaustion; cumulative attempts and infrastructure telemetry remain visible. Verified successes from any epoch remain successful. |
| `next_wave` | Existing workspace/family balancing consumes the current-epoch attempt count. No new scheduling policy. |
| Resume | Archive existing historical captures with their original plan; never launch an unstarted historical plan against a newer release. A new release gets a new numbered job. |
| Remote archive / export registry | Preserve raw receipts unchanged and archive the immutable plan/execution as `wave.json`, included in the produced dataset manifest. |

No old receipt, job, capture or verified success is reclassified or discarded. An immutable record may be reused identically; conflicting replacement fails. Status and the current terminal index are mutable summaries and explicitly name their epoch.

Validation covers historical exhaustion, current-epoch limits, infrastructure attempts, retained verified successes, actual wave scheduling after a release change, immutable original records, unfinished capture recovery, exact release/code provenance and restart binding. Tests mock collection and transfer; they perform no teacher inference.
