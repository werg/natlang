---
description: Choose what to do now and next from the diagnoses, the inbox obligations and the machine's resources.
args:
  diagnoses: RunDiagnosis[]
  triage: InboxTriage
  resources: Resources
  next_steps: Untrusted<string>[]
  allowlist: ActionSummary[]
  history: CycleMemory[]
  feedback: string
returns: Plan
---
You plan the next actions of the machine for the hourly check-in. allowlist lists every action that exists: its id, what it
does, the parameters it takes (name and kind of value) and the causes it answers (applies_to). Propose actions by their ids
from allowlist. A proposal is a suggestion that a crisp program checks; the owner of each action decides whether it runs.

Work in these steps:

1. Order the diagnoses: health "finished-failed" first, then "blocked", "gate-failed", "stalled", and the rest.
2. For each diagnosis whose health is not "progressing" or "finished-ok", pick the actions of allowlist whose applies_to
   contains its cause, and write one proposal per action. Put the run_id in `target` (or the unit name when the action takes a
   unit), fill `params` with every parameter the action takes, and write `why` as one sentence that cites the diagnosis. The
   first choice is the action that gathers more evidence. The next is the action that changes the run, such as a relaunch or
   a stop.
3. For each obligation in triage.obligations with reply_needed true, propose the reply action of allowlist, with the
   obligation's message_id as `target`.
4. Look at resources. When headroom_gb is large, gpu_utilization is low and teacher_load is low, a resource is idle: set
   `idle_resources` to its name (for example "gpu"), and propose the queue action of allowlist for the first step written in
   next_steps. Otherwise set `idle_resources` to the empty string.
5. history holds the last cycles with the proposals made and the actions the agents took. A proposal equal to one in the last
   cycle, when the run's health did not change since, gives way to the next action for the same cause.
6. Fill `cites` of every proposal with the run ids and message ids it answers, and `expected_effect` with what should change
   once it runs.
7. Write `summary` as at most three sentences for the status page: what runs, what failed, what is proposed.

A cycle with all runs healthy and no obligations has an empty list of proposals. When feedback is not empty, it lists what the
previous answer lacked; give an answer that supplies each listed point.
