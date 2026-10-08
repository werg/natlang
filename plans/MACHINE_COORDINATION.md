# Canonical checkouts, coordination and training-data synchronization

## Ownership

`/home/werg/natlang` is the sole active development checkout on each machine.
The Pop agent owns Pop GPU jobs, services and local training. The DGX agent owns
DGX GPU jobs, services and generation/training. Each agent can read the other
machine and transfer data/notes; execution changes on the other machine need
coordination or an explicit user request. This consolidation was user-requested.

The former `/home/werg/natlang-remote` is a historical alias into
`/home/werg/natlang/runs/dgx-legacy-imports/retired-root`; its campaigns, pinned
runtimes, worktrees and metadata are archived beneath the canonical repo. No new
jobs there. Registered archived worktrees remain only to preserve provenance and
uncommitted work, not as synchronization targets. All 14 inspected worktree HEADs
were already on main. The skill-build-v2 uncommitted export patch is preserved;
its main features are already in current main. Do not blindly reapply it.
Historical source/data references continue to resolve via aliases. Root-owned
model-download folders moved with their enclosing directory, requiring no sudo.

## Inbox

Each checkout has its own **gitignored** `.coordination/inbox.md`. It is a
scratchpad for coordination, never a data-admission authority or secret store.
Initialize/read on the receiving machine:

```sh
python3 scripts/coordination_inbox.py init
python3 scripts/coordination_inbox.py check --ack
```

Append a note from Pop to DGX using stdin (no interpolation of the note in shell
commands):

```sh
ssh dgx 'python3 /home/werg/natlang/scripts/coordination_inbox.py post --sender pop-agent' < note.txt
```

The DGX agent uses the same command with the owner's supplied Pop SSH alias and
`--sender dgx-agent`. The user will supply access; do not invent credentials.
Posting is locked to prevent lost concurrent appends. Reading with `--ack`
records a local byte offset; notes stay in the file. Check at session start,
before resource changes, and every monitoring cycle (at least every 50 minutes
while supervising work). Inbox files are independent: never rsync one over the
other. A timer checking a file would not wake a model session; the agents must
perform the read as part of their actual work cycle.

Useful notes: current job/unit + output path + source commit; planned resource
changes; published corpus IDs; admissions/holds; data that needs transformation;
conflicts or decisions the other owner should handle. Reply with what was acted
on and what remains. Durable decisions belong in Git/handover, not just inboxes.

## Code and pipeline additions

1. Fetch main before work; merge frequently. Commit small coherent changes and
   push `HEAD:main` (especially from a detached worktree). Handle conflicts rather
   than overwriting another agent's checkout. Keep an active run's loaded source
   pinned; synchronize code once it finishes or via a separate local worktree.
2. Add/modify converters, loaders, admission rules and recipes **in Git**. Use
   repo-relative paths or declared dataset roots, not private mirror paths.
3. For every new output, register a unique immutable ID, owner (`pop` or `dgx`),
   canonical repo path, selected files and admission status in
   `training/neuralese_corpora.json`. Identify source snapshot IDs, converter
   commit/options and exclusions in the output's own provenance manifest. Include
   pieces tables, dependency/producer records, negatives, split/admission reports
   and rejection evidence needed to interpret it. Avoid dangling call graphs.
4. Publish on its owner machine, then commit/push the registry and generated
   small manifest (not dataset bytes):

   ```sh
   python3 scripts/sync_training_corpora.py publish --machine dgx --id NEW_ID
   git add training/neuralese_corpora.json training/corpus-manifests/NEW_ID.json
   git commit -m 'Register sealed NEW_ID corpus and its provenance'
   git push origin HEAD:main
   ```

   Never mutate a published ID. New generations, repairs or conversions get new
   IDs with explicit replacement/derivation relations. Active collectors keep
   writing their run directory; publish a frozen output only after finalization.
5. The receiver pulls main and synchronizes the exact snapshot:

   ```sh
   # From Pop: DGX-owned snapshots pull; Pop-owned snapshots push.
   python3 scripts/sync_training_corpora.py sync --machine pop --host dgx --id NEW_ID
   python3 scripts/sync_training_corpora.py status --machine pop
   ```

   From DGX use `--machine dgx --host POP_SSH_ALIAS`. Transfer includes only
   manifest files, rejects destination content collisions, never deletes, checks
   space before copying and writes SHA-256 verification receipts beneath
   `.coordination/corpus-receipts/`. Interrupted partials can resume. An explicit
   `verify` reads local bytes again; `status` only reports local presence, not hash verification.
   `--machine` identifies the machine running the CLI; it never selects a remote
   verification destination. Use `verify-remote --machine pop --host dgx --id ID`
   to hash actual DGX files. Verification receipts name their scope, hostname and
   repository. Before removing a local artifact copy, independently verify the
   remote exact manifest and its hashes over SSH, as well as local hashes and
   live references. A local receipt is never backup evidence. Stop if any remote
   file is missing; preserve failed verification and unresolved loss explicitly.
   For recoverable local storage offload, use the manifest-driven command
   rather than manually sequencing verification and `unlink`. `--machine`
   names the local owner; `--host` names the remote destination. Pop defaults
   to DGX, while DGX should pass the Pop host explicitly:

   ```sh
   # Read-only preflight: exact manifest paths, local hashes, open FDs and
   # visible live-process references; prints the matching execute command.
   python3 scripts/sync_training_corpora.py offload --machine pop --host dgx \
     --id ID --file checkpoint.pt

   # Executes only after review: transfer the selected manifest files, verify
   # them on the actual SSH host, recheck local bytes/references, then unlink.
   python3 scripts/sync_training_corpora.py offload --machine pop --host dgx \
     --id ID --file checkpoint.pt --execute
   ```

   Offload requires the selected committed registry entry and immutable
   manifest; unrelated registry additions do not block it. Each selected
   path must occur verbatim in that immutable manifest; globs and arbitrary
   paths are rejected. The execute command reuses exact manifest sync, writes a
   remote subset verification receipt without replacing the full-corpus receipt,
   then writes `.coordination/artifact-evictions/` with the remote hostname/repo,
   local checks, selected files, and a restore command. Unrelated inaccessible
   `/proc` FD directories are recorded, not treated as global blockers; an
   observed selected-file FD or live process reference stops before unlink.
   A matching Linux kernel boot ID on both sides stops unlinking even if host
   names differ; duplicate host names alone do not.
   Restore the selected paths with the command in the receipt. This is storage
   management only; it changes no corpus admission or training qualification.
6. Announce snapshot IDs, source coverage, counts and admission state in the
   receiving inbox. Update the dataset coverage/handover. Train only after the
   normal source-policy, split/protected, native replay and quality gates.

### Long-lived local dispatchers

When a reviewed queue needs to outlive a shell command, do not assume that
`nohup ... &` or an `&`-terminated shell command leaves a durable worker behind.
In the Pop Codex shell environment, a dispatcher started that way exited before
recording `campaign_open`; its launch log was empty and no claim, case runner or
provider request was observed. A bounded marker probe showed the same pattern:
an immediate marker could be written, but a delayed marker and a SIGTERM-handler
marker were absent after the command returned. This establishes that the
background child did not survive the tool invocation, but does not establish
whether it received a signal or what its exit status was.

For an authorized long-lived dispatcher, use `subprocess.Popen` with
`start_new_session=True`, a persistent log file, and immediately record the
actual PID, argv, working directory, plan hash and process-group/session IDs.
Wait for the dispatcher’s durable `campaign_open`/claim handshake before
reporting it as active. At each monitoring cycle, reconcile those records with
the live PID/argv and claim ledger; a launcher PID by itself is not evidence of
an active campaign. Preserve failed launch-attempt logs and status uncertainty
instead of replacing them with a later successful attempt.

There is no broad mutable two-way rsync. The former
`natlang-dgx-development-data-sync.service` on Pop was stopped and disabled
because it pulled only one DGX namespace and could miss new outputs elsewhere.
Existing mirrored inputs remain available. Do not restart that old writer.
Both agents explicitly run the snapshot sync during each monitoring cycle after
pulling new manifests; they should own the artifacts they publish. Large files
can remain on the DGX external drive or internal dataset volume via canonical
repo symlinks; links are storage placement, not separate development checkouts.

## Current corpus and integration gaps

See `training/neuralese_corpora.json` and `training/neuralese_dataset_coverage.json`.
The source-level inventory remains `training/neuralese_data_ledger.json`; the
new registry records concrete current outputs rather than replacing that ledger.

- **S1:** 1,870,591 records across 45 family files (1,361,548 train, 52,439
  validation, 456,604 test), about 68 GiB. Current schema/structural audit passed
  with zero errors/duplicate IDs. Admission still requires source-policy and
  ambiguous-alias/protected split review; it is not all ready for training.
- **S3 subset:** 45-family derived training selection (~21 GiB); inherits S1
  holds and is not an independent additional corpus.
- **v13→neuralese v8:** 111,301 records (~7.6 GiB), plus pieces. Converts prompt,
  instruction and compaction sites. Only 153 child-result writers/342 reads;
  1,693 missing producers and 476 unprinted values remain exact. We need richer
  recurrence examples and explicit current admission, not just more copies.
- **S2:** generated crisp-skill, adversarial, optimization/semantic families and
  paired search/repair evidence are retained under `runs/dgx-development-generated`.
  Mixed candidates and quarantined attempts must be passed through the paired
  improvement/transfer gates and converted to current neuralese records.
- **Maple:** v13-r2 native SFT is a rendering of existing examples, not new source
  coverage. `maple-nested-20261005/mixed-v1.jsonl` is a separate broad-domain input
  and is registered independently. Expert slice benchmark files are model
  measurements, not training examples.

No aggregate "ready total" adds these overlapping variants together. The
55 migrated source entries are not proof that every family has current, fully
admitted neuralese trajectories; 22 queued and 58 source-only entries still need
explicit converter/coverage decisions. The DGX owner should inventory any new
combinator/teacher-generation output and register it; this sweep found converted
v8 and generated S2 artifacts, not a new broadly populated recurrence corpus.

### Consolidation receipts and newly visible backlog

The current Pop snapshots, including the 68-GiB full S1 candidate and 10.34-GB
historical teacher evidence, are hash-verified; `.coordination/corpus-receipts/`
holds machine-local receipts. The native corpus and selected student adapter are
also verified on DGX. S2 v2 is materialized as independent bytes so further run
updates cannot mutate that snapshot. Retired S2 v1 remains an explicit replacement
record, excluded from default transfers.

Historical campaigns are integrated as `teacher-campaign-evidence-20261005`.
`training/audits/teacher-coverage-20261005.json` flags 8,813 accepted historical
IDs not present in the native corpus identity index. Resolve alias/version,
cutoff/import and protected/split decisions before counting or admitting them.
The DGX owner should take this queue, publish a refreshed native snapshot, then
publish its modern neuralese conversion. `audit_teacher_corpus_coverage.py` is
reusable for subsequent cutoffs. A historical folder is not a disposition: useful
outputs must enter a recipe or a specific transformation/admission queue.

The existing general inventory now shows registered neuralese snapshots by their
manifest, rather than silently missing symlink-backed data or repeatedly hashing
the whole S1 corpus through an unrelated native-SFT builder. Representation-specific
loaders remain necessary; replication is not automatic format conversion.

Completed local diagnostics are also registered: `local-recurrence-quality-v1-20261005`,
`local-recurrence-inputs-20261005` and `local-recurrence-stress-cohort-v1-20261005`.
The first includes the full optimizer/RNG checkpoint, heads, soft parameters and
evaluation report; the other two pin its inputs. These are diagnostic artifacts,
not a training-data admission or a claim about the best SFT student. Shared head
and control parameters are now being tested with soft-context updates disabled on
Pop; DGX resource management remains with the DGX agent.

## Bounded test scratch on shared training machines

Run focused container tests with a read-only root and repository mount, a bounded `/tmp` tmpfs (512MiB by default), offline model access and cache paths under that tmpfs. CUDA/Triton JIT shared libraries require `exec` on the bounded tmpfs (`/tmp:rw,exec,nosuid,size=512m`); the Docker default `noexec` causes a deployment failure before the first update. Broad test discovery can create gigabytes of disposable files even with a read-only repository. An incomplete suite does not qualify a change. Monitor host disk headroom before tests and checkpoints; preserve active parent/full optimizer checkpoints. Offload inactive local weights only after fresh local and canonical remote SHA/size verification, recording the receipt and notifying the owning agent. Never remove the sole canonical backup.
