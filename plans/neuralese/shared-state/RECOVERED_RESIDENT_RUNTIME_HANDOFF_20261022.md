# Recovered resident runtime handoff

The supervisor chooses runtime and recipe changes to keep useful work advancing.
The existing handoff machinery remains the exact mechanism for preserving owners,
checkpoint state, source history and scheduling debt. No model-facing policy changes.

A recovered learner can execute newer qualified code while carrying the last
adopted plan produced by its historical reader. These are distinct facts. The
historical package verifies the old plan; the captured executed package verifies
the actual recovered learner. Neither package may impersonate the other.

1. Drain the actual captured recovered optimizer at an update boundary and retain
   its full optimizer, RNG and per-row exposure. Verify actual owners have exited.
2. Bind the immutable historical plan package and the actual captured executed
   package separately. For a differing executed package, verify the saved passed
   memory qualification, its actual launch argument, parent, code and capacity
   certificate. This does not grant task-quality or recipe-completion credit.
3. Apply existing recipe lineage checks using the historical plan's exact reader.
   Preserve historical recipe, source, classifier and adoption identities.
4. Rejoin with an unused operation namespace, preserving earned state ahead of
   stale bridge results. A recovered launch may start from an already adopted
   generation; verify that generation against its actual full parent rather than
   relabeling the original resident adoption as a new one.
5. Run the entire evolving schedule, including source additions, preference and
   whole-graph RL. A passed handoff is operational evidence, not trained-model lift.

Exact verification and transport stay in shared Python code because they compare
bytes, tensor/state identities and actual OS process records. Scheduling decisions
remain the supervisor's natural-language policy above.

Implementation status at 2026-10-22 16:44 UTC: the shared handoff verifies both
scalar and complete cohort qualification evidence, including zero optimizer
updates during qualification, passed numerical references, successful whole-graph
capacity cases and measured certificate limits. Historical readers and executed
packages stay separate. Observed bridge capture records its actual observation
time and explicitly does not claim startup capture.

Independent Luna engineering review identified a missing numerical/capacity
content check in the scalar path. Both formats now use the same content verifier;
negative scalar cases cover nonzero updates, boolean update counts, failed
references/capacity/cases, empty cases, invalid sizes and inflated limits. The
focused ledger unit `recovered-runtime-independent-review-repair-20261022`
passed 54 tests. Earlier integration checks include actual next updates in both
model fixtures. A read-only probe also verified the live Mellum recovery's real
saved cohort and historical reader, without signalling the learner.

Canonical production rejoin is still incomplete. In particular, the production
caller must capture/drain the actual bridge, authenticate an already-adopted
resume generation against its real parent, reserve unused operation work and
load these controls after every sealed loader. Fixture next-update success and
the read-only cloud probe do not establish those production integrations. Keep
the existing useful recovery learner running while completing them.
