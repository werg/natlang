/** `natlang help traces` and `natlang traces --help`: kept apart from the command so help never opens the store. */
export const TRACES_HELP = `The machine's record of natlang calls (NATLANG_CALL_STORE, default ~/.local/share/natlang/calls):
  natlang traces list [--program DIR | --all] [--definition NAME] [--executor agent|crisp|crisp-agent]
                      [--outcome done|failed|...] [--since ISO|30m|2h|7d] [--limit N] [--after CALL] [--audits]
                                             Recent calls, newest first. Inside a program: that program's calls;
                                             --program DIR another program's; --all every call on the machine.
  natlang traces show CALL [--full] [--events] [--json]
                                             What the model did, turn by turn: its reasoning, text, tool calls with
                                             their results or rejections, child calls and the outcome. --full shows
                                             long texts whole; --events prints the raw trace events (JSONL); --json
                                             the record with its values.
  natlang traces good CALL                   Tester verdict: annotate, and save the call's inputs and output as an
                                             example in the program's examples.json (natlang test checks it,
                                             natlang tune learns from it).
  natlang traces bad CALL [--expected JSON]  Annotate; with --expected also save the example (plain bad saves none).
  natlang traces fix CALL --expected JSON    Annotate and save the example with the right answer (replaces the
                                             expected result of an earlier example with the same inputs).
    good|bad|fix options: --group G (else NATLANG_FEEDBACK_GROUP, else feedback-<day>-<host/user hash>: one
    tester's calls share a split side), --program DIR (default the call's recorded program, else the program of
    the current directory), --dry-run. Only root calls of a program .nl function become examples; others keep the
    annotation and say why.
  natlang traces status                      Store size, settings, pending shadow/audit jobs.
  natlang traces hot [--program DIR] [--since ISO] [--by calls|tokens|wall_ms] [--limit N]
  natlang traces occupancy [--definition NAME] [--limit N]   Batch occupancy: requests in flight, batch size, queue wait.
  natlang traces tiers [--definition NAME]                   Tiered engine: calls per tier, deopt rate, cost per tier.
  natlang traces export [--definition NAME] [--limit N]      One JSON record per line, values inlined.
  natlang traces pin|unpin CALL              Keep a call when the store evicts.
  natlang traces annotate CALL KIND VALUE    Attach feedback or a judgment to a call.
  natlang traces config [KEY=VALUE...]       maxValueBytes, maxStoreBytes, specialization (off|shadow|on),
                                             auditRate, acceptanceBound, promotionComparisons, minCalls.
  natlang traces evict [--bytes N] [--work-limit N]
CALL is a call ID, a unique prefix of one, or the same without its task-N- prefix (the first 8 characters
usually do). Every command takes --store DIR to read another store.
Every runtime records each call (exact inputs, output, service calls, eval code, cost) unless
NATLANG_CALL_STORE=off. A program's natlang.json can exclude values: "recording": { "exclude": [...] }.`;
