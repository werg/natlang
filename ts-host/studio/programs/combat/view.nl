---
args:
  state: State
returns: View
---
Describe the current Sparring grounds workspace for its user.
Return heading (a short useful title), summary (one sentence grounded in state),
focus (the panel IDs in the order most useful now), and suggestions (up to three
short optional next-step commands). Available panels: arena, moves, rounds.
Include each panel exactly once. Never claim an operation succeeded unless the
state contains its result. Distinguish semantic proposals, simulated outcomes,
and verified execution receipts. Keep the language warm, concrete, and brief.
