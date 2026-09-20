# Log investigator

`step.nl` folds incoming log records and source-gap events into an incident
state. Natlang assesses significance using a bounded, exact evidence query;
crisp helpers recheck source identity, count distinct corroborating records,
track cursors and guard an optional alert sink. A log line is evidence data,
never an instruction to the interpreter. Benign bursts can be ignored; a gap
remains visible in the state's unknowns.

`LogWorkspace` retains the indexed lines and notification receipts in the
native host. It records event time and arrival time separately, supports
out-of-order observations, and queries a declared time window by service/code.
An escalation requires at least three distinct matching source records and
the supplied evidence must exactly match those records. A stable incident key
suppresses duplicate notifications across nearby events. Sink errors produce
an `unknown` receipt rather than an assertion that the notification failed or
never happened. The sink is optional; no external notification is sent by the
application tests.

The Fold integration test replays a regression burst, duplicate delivery, a
source gap and misleading benign log text. A separate test rejects fabricated
evidence and checks uncertain alert delivery. The model callback is scripted;
independent incident-quality review and live-model behavior remain open.

Production gates are bounded source buffering, persistent cursors/receipts,
reconciliation of unknown deliveries after restart, richer query choices,
and calibrated false-positive/false-negative evaluation. These are application
host facilities and do not require a global search or interruption primitive.
