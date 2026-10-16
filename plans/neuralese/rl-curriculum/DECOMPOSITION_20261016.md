# RL task-curriculum metadata update (2026-10-16)

The current 128-row RL task source remains intact as a complete breadth layer. This update adds scheduling metadata only; it does not change prompts, inputs, source identities, expected outputs, tools, or execution oracles.

1. Read the integrity-bound v1 train task manifest and its already-reviewed base/broad selection, source, and protection proofs.
2. Select lower levels only by exact task identity plus structural assertions over the existing oracle: the supplied-integer/no-work skill-routing case is level 0; the other three single-call skill-routing cases are level 1; the eight one-record port-record cases with exactly one child result are level 2. All other rows remain level 3.
3. Copy the original row and add `curriculum.rl.level` and a concise scheduling reason. Never duplicate or rewrite a task to manufacture an easier case.
4. Recheck all rows remain train-only and unique; remove the new scheduling field and require byte-canonical equality with all 128 original task payloads. Revalidate current decision holds and source-group protection.
5. Write the new union and audit durably, bind both hashes in a new task-source manifest, and register the immutable files as a new version derived from v1.

The level assignment is a scheduling prior, not an RL eligibility rule or a claim that more complex tasks are unusable. Policy stays in this reviewed exact-ID map; the script is only deterministic selection, hashing, and verification plumbing.
