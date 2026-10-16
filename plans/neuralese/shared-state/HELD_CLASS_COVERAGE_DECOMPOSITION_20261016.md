# Native held monitoring coverage

The monitoring policy is: preserve previous held exposure, check ordinary and
genuine communication graphs during each bounded normal evaluation, and check
the full held source at final qualification. This changes monitoring selection,
not training examples, weights, optimizer state or quality requirements.

1. Read authenticated prepared source entries and classify each existing graph
   using its communication field. Do not parse token payloads again.
2. Recover legacy visits from the recorded source size, completed rotations and
   cursor. Preserve each class's exact visit counters thereafter. A contradictory
   historical rotation needs diagnosis rather than an invented coverage reset.
3. Select a whole graph from each available class, then alternate within the
   existing token/graph budget. Prefer less visited rows and preserve class-local
   cursor order. An indivisible class minimum may exceed a small budget; record it.
4. Evaluate selected graphs in ascending source order, avoiding repeated source
   decompression. Record selected IDs, source binding and both class cursors.
5. At final qualification evaluate every held row. Append-only held additions
   retain previous per-row visits and prioritize unseen rows before repeats.

`train/native_held.py` is the shared exact index/counter mechanism for this
owner-declared policy. The existing native SFT evaluator performs the actual
forward/readout once per selected graph and reports boundary telemetry. Both
ports use the same selector. Existing runtime/task qualification remains separate.
