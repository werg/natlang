# Fill homogeneous campaign waves

Inventory before edits: repository-wide `next_wave(` search finds one scheduler
(`scripts/neuralese_data/vibestudio_campaign.py:532`), one driver call (674),
and direct campaign tests. `collection_limits` separately caps actual workers,
including nested deployment limits; it does not select tasks. `publish_status`
provides pending/exposure/attempt ranks. Saved immutable wave plans resume their
original cases and bypass new selection. Other engines, training, held evaluation,
IR admission and archive tools do not schedule these workspace/review groups.

Actual V29 selected only two Spectrolite tasks despite eight configured workers
and1498 pending goals: selection first picked the globally lowest rank/fairness
case and then required its exact rank and workspace/review group. Prefer, among
groups containing the best pending primary exposure band, a group with enough tasks in the same
primary exposure band to fill configured workers. When no group fills workers, prefer the largest available tail. Preserve exact
rank and workspace attempt fairness among equally fillable groups. Within the selected group, draw exact
ranks in order, rotating families within each rank, and use later exposure ranks only until worker slots are filled, rather than
padding productive waves with retries merely to reach the wave-size cap. A wholly retried group cannot displace a group containing fresh goals.
Later ranks never displace earlier ranks inside the chosen group. All tasks
remain pending until selected and retain their original identity and retry cap.
`--strict-priority-waves` restores the previous exact-rank group policy.
