# Current-owner checkpoint import

## Natural-language policy

The existing recipe owner decides when an in-process trainer may yield and
which already-saved checkpoint to resume. Importing a checkpoint never earns a
RAW round, completes a recipe stage, or qualifies communication readiness. It
only makes an explicitly selected, already-loaded full training state the
current checkpoint for the existing recipe cursor. Normal `student_cycle`
continues from the preserved phase, pending round, stage index, cycle number,
and all outstanding task/source obligations.

An owner may select an older saved checkpoint only as an explicit supervised
recovery decision. The decision records its reason and both the current and
selected checkpoint bindings; it does not rewrite ancestry or claim that later
work was completed. A sealed RAW learner, pending round, or active central
operation cannot be bypassed by this import.

## Exact mechanism

1. The in-process owner supplies a sealed authority with schema
   `natlang.native-raw-current-owner/2`. It separates `process_argv`, the live
   `/proc` vector, from the in-process trainer command. It binds current
   PID/boot/start ticks and argv bytes; wrapper path/SHA; RAW config path/SHA;
   installed package path and complete code inventory; the final gate-spec
   path/SHA; the gate spec's `effective_trainer_argv` binding to the serialized
   argument list; and a separate binding to the serialized full command. The
   importer verifies the gate spec and argument list agree, then checks the
   full command is exactly `[sys.executable, '-u', '-m',
   'natlang_neuralese.train.native_state_sft', *arguments]`, with its output
   and selected checkpoint matching the prepared RAW continuation. It also
   binds selected checkpoint path/SHA/byte count and a saved-state summary
   binding that checkpoint to global step and model identity. The caller
   passes the saved-state object already loaded by the owner; import performs
   no model or checkpoint decode.
2. Under the existing controller and production locks, re-read and bind the
   configuration, current state, RAW cursor, owner identity, and selected
   checkpoint bytes. Check that the saved-state global step and training
   identity agree with the selected checkpoint authority.
3. Reject an active non-collect production operation, an active sealed RAW
   learner, or any uncompleted `pending_rl_round`. Require a recovery reason
   whenever the selected saved step is earlier than the current published step.
4. Build durable before-snapshots and candidate JSON files. Change only the
   current checkpoint references and the established pending-load/readiness
   fields. Preserve every other cursor and state field byte-semantically,
   including RAW phase, collection cursor, task cases, stage debt, cycle and
   exposure. Do not mark any operation, stage, or recipe complete.
5. Acquire the existing RAW controller and production-controller leases in
   stable path order. The RAW controller lock is `controller.lock` beside the
   per-model RAW directory; the production lock is inside the RAW output
   directory. If passed the inherited canonical controller descriptor, verify
   its inode and existing exclusive flock without acquiring or unlocking it.
6. Write a durable intent binding the exact request and candidates, then
   compare-and-swap only from the captured current state/cursor bindings. An
   exact retry reconciles the same intent; an unknown successor fails closed.
   The ordinary recipe controller then resumes through its existing
   `student_cycle` and RAW/supervised operations.

## Crisp-code boundary

PID/boot/start-time checks, file hashes, lock acquisition, durable snapshots,
and atomic compare-and-swap are exact external-world custody and durability
mechanisms. They do not select policy, task difficulty, or stage completion.
The natural-language recipe and owner retain those decisions.

## Required checks

Focused protocol tests bind a real in-process-shaped owner authority and verify
preservation of unrelated RAW/cursor debt, explicit older-checkpoint recovery,
rejection of missing recovery reasons and pending/active operations, rejection
of owner/config/package/checkpoint substitution, exact crash-retry behavior,
and refusal to overwrite an unknown successor. Tests must establish that no
recipe completion, exposure, optimizer, or readiness credit is manufactured.
