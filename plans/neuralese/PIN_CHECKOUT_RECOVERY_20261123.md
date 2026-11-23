# Interrupted pin creation recovery

Inventory before changes:
- `scripts/pin_checkout.py:create/main` owns experiment pin worktree creation,
  ignored-state copying, build and completion marker; CLI remove owns deletion.
- `scripts/neuralese_data/expert_iteration.py:1008-1016` invokes this same CLI
  through a resumable ledger job. An interrupted add/build must be retryable.
- `scripts/neuralese_data/grpo.py:274-286` consumes the ready marker and commit;
  `training/neuralese/natlang_neuralese/common/data_registry.py:68-80` consumes
  its source path. The marker shape stays unchanged and is published last.
- `scripts/full_check_service.py:77,191,227` and
  `scripts/neuralese_data/check_sglang_hybrid_swa.py:300` create/remove their own
  check/third-party worktrees, not experiment pins; no pin marker/copy/build
  mechanism is used there, so their lifecycle is unchanged.
- `tests/neuralese/test_pin_checkout_shared.py` exercises the canonical helper.

Change: serialize create/remove by canonical destination under Git's shared
common directory. A missing marker can resume only the exact registered detached
worktree at the requested full commit, with no modified tracked sources. Reject
foreign directories, symlink destinations and mismatched commits, including when
an existing marker claims readiness. Ignored-state copying and shared links must
be repeatable. Rebuild and revalidate before fsync + atomic completion publication.
The lock is inherited by preparation/build children so killing their parent cannot
let another helper modify/remove a still-building pin. No active pins are edited.

DGX retry exposed a second interrupted-state case: a correct preexisting link
whose canonical source itself is dangling has exists()==false. All LINKED leaf/
root and SHARED link creation now uses one non-dereferencing lexical target join.
Correct dangling links survive retry; foreign targets and conflicting files or
directories fail without replacement. Tracked runs/data root directories remain
containers for their ignored child links. Actual failed-build/retry tests cover
both dangling leaves, a dangling venv root, and a dangling native-model store.
17 targeted tests and the ledgered quick gate passed; no active pin was touched.

Nested ignored environment follow-up: Git can emit both a directory and its
children when nested .gitignore rules differ. Prepare shared roots first and
skip every covered descendant; this also removes redundant ignored-state copies.
A prior helper-created directory is reusable only as an exact canonical symlink
view; missing aliases are completed, unique files are rejected without deletion.
Shared model roots are registered before any copying, excluded from every
ancestor rsync and never copied by exact/descendant entries. Ordinary source
symlinks are linked without dereferencing; rsync never receives a symlink target
as its destination. A realistic nested-venv/model-ignore failed-build retry
regression preserves canonical model inode/content/ctime/mtime.19 tests and
ledgered quick gate passed.
