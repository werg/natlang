# LFM execution-recovery TRAIN batch (2026-10-19)

## Natlang-native design

This batch adds independent TRAIN program inputs, not teacher outputs. Each program is a natural-language task over a small, newly constructed service world. The source builder supplies a complete executable reference and exact result oracle; the existing Qwen teacher queue samples the responses later. No held input, answer, response or trace enters construction.

The ordinary track uses the existing `natlang.program/2` and `teacher_source_sequence` contracts. Its root reads both pages, asks the typed `select_record` function about every row, retains the selected IDs and values, writes one typed summary, reads it back, and returns it. Each world is small enough for a short complete trajectory, spans all eight existing task domains, and includes independent seeded runtime feedback variants for a bad page range, a reserved service binding, and a mismatched typed result. Exact JavaScript service boundaries and the Python source/oracle verifier are crisp because they describe outside-world behavior and exact grading; task choice and recovery remain natural-language policy.

The communication track uses the existing authored-root trajectory contract and `nl.with` typed private state. A sampled producer writes an evidence string; a sampled consumer reads that actual state and returns the exact requested public value. The reference requires the full producer/consumer call sequence. These are candidate graph tasks; only captured teacher trajectories with actual writer-to-reader state edges can count as complete graphs. They are not synthetic graph claims or a new transport implementation.

## Existing mechanism mapping

- Fresh world facts and complete references: `scripts/neuralese_data/build_fresh_recovery_teacher_worlds.py` plus one sibling profile builder.
- Exact reference qualification and protection closure: existing current-protection teacher reference qualifier.
- Actual sampled responses: existing Qwen source sequence and guardian, owned by the generation queue owner.
- Per-row quality/admission and split decisions: existing teacher trajectory review and `teacher_source_sequence`; reference qualification alone never admits training.
- Training: existing shared RL/RFT and native whole-trajectory mechanisms; no new optimizer, reward, queue, server or scheduler.

## Validation and release boundary

Tests must verify deterministic distinct IDs, all-train rows, current held/quality/pending exclusion, short page coverage, exact oracle equality, seeded failure recovery requirements, and graph contracts with producer/consumer edges. Run the existing exact authored-reference qualification on the generated packet before handing off. Teacher execution, teacher acceptance, training admission and actual student exposure remain separate milestones.
