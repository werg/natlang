# Relative training continuation quanta

The supervisor's existing natural-language policy chooses when to refresh and
which production work to run. A declared supervised quantum means that many
new optimizer updates from the restored stage cursor, unless genuine source
completion, qualification or an operational stop ends the work earlier.

1. The shared controller translates the declared quantum into an explicit
   relative update count in both CE and offline preference trainer arguments.
2. The shared native trainer counts actual successful updates since restoration.
   Periodic durable checkpoints keep their independent absolute/time cadence.
   At the relative boundary it saves a full checkpoint even off that cadence.
3. The existing report records actual updates and continuation status. Optimizer,
   RNG, source exposure and whole-recipe completion retain their existing rules.
4. Without the new explicit option, legacy refresh-at-next-checkpoint behavior
   stays available. The option is operational scheduling, not an objective change.

These are crisp argument validation, integer accounting and durable checkpoint
mechanisms. They introduce no model-facing prompt or new scheduling policy.
