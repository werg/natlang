# Resume evaluation workers independently after disk recovery

Actual LFM12964 protected evaluation had worker00 decoding while workers01–07
had stopped before any case at the 20480MiB floor. Root storage recovered, but
`wait_collectors` first blocked on every child's `wait()` in array order. It
therefore could not restart the seven paused workers until the busy first
worker finished. This is an orchestration bug, not model/task failure.

The existing task evaluation is an exact execution verifier. Keep its policy,
packet, actor, task budgets and scoring unchanged. The crisp process mechanism
polls each owned collector, closes its terminal log, validates an exact
before-case storage pause against the immutable queue snapshot and journal,
and relaunches that collector independently when its disk floor is satisfied.
Running collectors continue. Operator drains, started cases, changed queues and
mismatched floors retain their existing rejection behavior. No new component or
model-facing text is introduced.

Regression coverage makes the first child remain busy until a second paused
child is restarted; a blocking wait is forbidden. Existing recovery and negative
identity tests remain required. Both student evaluators share this mechanism.
A running Python frame cannot adopt this change automatically: any interrupted
old attempt must be recorded as unscored, retain its diagnostic partial only for
the assigned bug investigation, and retry the unchanged actor/request through
the existing dispatcher after the fix is checked. Never fabricate a score.
