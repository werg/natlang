# Log Console

An incident investigator over a stream of log events (guided or stdin JSONL). See `DECOMPOSITION.md`.

- `investigate.nl` and `investigate/`: the investigation in natural language. Significance (pluggable crisp or natlang),
  incident clustering and folding, hypotheses, evidence queries planned over the index, weighing, the escalation
  decision with its evidence rule, severity, summaries, gaps.
- `index.ts`: the exact log index (`LogIndex`) and the idempotent alert sink (`AlertSinkService`) as services, the
  pure `commit` of a decision into the incident state, and `step`.
- `console.ts`: the terminal and stream CLI.

Hot path: `LogSettings.significance` is `"natlang"` (default) or `"crisp"`.

Tests: `ts-host/test/log-investigator.test.mjs` (scripted models). Check: `natlang check applications/logs`.
