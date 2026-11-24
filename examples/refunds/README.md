# Refund desk

A refund desk for several shops. TypeScript holds the orders, the policies, the floor and the ledger; every judgment
is a typed natural-language function.

```sh
natlang run examples/refunds                                   # settle the bundled requests.json
natlang run examples/refunds -- --eligibility shadow           # both eligibility sides, disagreements traced
natlang run examples/refunds -- my-requests.json              # your own JSON list of requests
natlang test examples/refunds                                  # the eligibility cases in examples.json
natlang traces show CALL_ID                                    # what one call did
```

## Files

- **types.ts**: the data. A request's `text` is `Untrusted<string>` (the model reads it as quoted data); a `Reply` is
  `Is<string, "...">`, a property the runtime checks on every reply.
- **eligible.nl**: `(request) => boolean` with `readout: decision`, the judgment that also reads what the customer
  reports (damaged, not what they ordered).
- **verdict.nl**: `(request) => Verdict` with `readout: decision`. Its instructions name `policy`, which comes from
  the context it is bound to.
- **rules.ts**: eligibility as `pluggable({ crisp, nl }, setting)`: the 30-day return window on the order record, or the
  judgment. `shadow` runs both, serves the rule and records every disagreement as a `pluggable_shadow` trace event.
- **desk.ts**: `settle`, one independent pipeline per request, all at once. The verdict is wrapped with
  `escalate(verdict.in(Context.empty().with({ policy })), { to: 'teacher', floor: 0.8 })` from `@natlang/node/std`:
  the same function bound to each shop's policy, scored locally and escalated only below the confidence floor (0.8).
- **reply.ts**: an inline `nl` writes the reply, and `iterateOn(revise, first).until(nl\`reply answers every question
  the customer asks in request.\`)` revises it until the natural-language check holds.
- **ledger.ts**: records the outcomes as one folder transaction: `folder.propose(reducer)` writes a case file per
  request and updates `ledger.json` on a private copy; the program checks that the proposal changed only those
  files, then `folder.accept(proposal)`.
- **main.ts**: the `desk` target. The ledger lives in the program's state directory.
- **policies/**: each shop's written refund policy; **requests.json**: the sample requests.

`teacher` is the profile that your model profile names as `escalate` (a stronger hosted or local model); without one,
`escalate` keeps the local answer (`escalated: false`, `reason: 'unavailable:...'`) instead of rerunning on the
default model. `natlang run examples/refunds -- --eligibility shadow` runs both eligibility implementations and traces
where they disagree.
