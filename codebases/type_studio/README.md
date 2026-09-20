# Natlang type studio

`infer.nl` proposes a TS-style signature from a bounded source snapshot and caller
obligations. The exact host parses each proposed type, checks declared named types,
checks caller and return fit, and verifies required effects. `consistent` means the
proposal fits known evidence; it is not a proof for every semantic execution path.

`check.nl` asks natlang to identify possible calls. A diagnostic is `exact` only when
the claim matches a frozen execution witness supplied by the host. Other paths remain
`hypothesis` or `unknown`. Missing signatures are analysed as source data; they are not
installed into a checked running graph. Source revisions accompany every view.
