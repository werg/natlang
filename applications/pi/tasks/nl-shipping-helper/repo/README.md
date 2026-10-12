# shipping

A natlang program. `quote.nl` is a natural-language function: the model follows its numbered
steps. Steps that need exact arithmetic call crisp helpers in `helpers/`, plain JavaScript that
must do exactly what the step says. Helpers are unit-tested with `npm test`.
