# DGX development checkout and pipeline data

The checkout is `/home/werg/natlang` on `ssh dgx`, matching the home machine's
repository path. Source is updated through Git fast-forwards, independently of
data replication. The initial setup updated the user's clean clone; subsequent
code synchronization must preserve development edits on either machine.

## Environment

- Node 24.0.1 ARM64 lives in `~/.local/share/natlang-dev/`, with commands in
  `~/.local/bin`. It is separate from sealed generation runtimes.
- `ripgrep` 14.1.0 ARM64 is installed in user tooling without replacing system packages.
- Pinned npm dependencies are installed with the workspace lockfile; Node and
  browser outputs are built. Python checkout tools use `~/natlang/.venv`.
- `~/.local/bin/natlang` points to this checkout's wrapper. `uv` is installed in
  the Python environment at version 0.9.26.
- The `dgx-qwen` profile uses the existing NVFP4 Qwen server on port 8082. It is
  externally owned; development commands do not launch another GPU server.

```bash
ssh dgx
cd ~/natlang
export PATH="$HOME/.local/bin:$PATH"
natlang doctor --json
npm run build
source .venv/bin/activate
```

The Python distribution is dependency metadata for checkout scripts. Explicit
empty package discovery prevents setuptools from treating source datasets,
`runs`, and project plans as importable Python packages.

## Replicated pipeline

Large untracked branches within `data/`, `runs/` and source-dataset directories
are linked into `/mnt/external/natlang-development-data`. Git-tracked parent
directories and files remain real checkout paths, so Git can inspect and update
them normally. `scripts/link_development_data.py` refreshes these links after
each copy and preserves pre-existing development files.

The mirror includes source corpora, unconverted source material, case IR,
static adapters' inputs/outputs, generated trajectories, failures and partials,
source-review decisions, replacement history, recipe inputs, admission/split
audits, rendered training data, and sealed runtime evidence. A copied candidate
or rejected trajectory remains a candidate or rejection; existence in the
checkout does not grant training admission.

Full run runtime dependencies/toolchains are retained for provenance. Archived
x64 binaries are not ARM execution entrypoints. The live DGX generation runtime
remains the separately sealed ARM runtime under `~/natlang-remote`; development
uses the ARM checkout dependencies.

Base-model downloads in the home machine's `models/` and unrelated Python
environments are omitted. Model revisions remain pinned in the pipeline plans;
use the existing DGX external model store when running model-dependent tools.
Credentials, locks and temporary files are excluded. Historical source and
review records needed for replacement lineage are retained.

## Ongoing synchronization and ownership

Service definitions are retained in `scripts/systemd/`. Install them in
`~/.config/systemd/user/` on the home machine and reload the user manager when
recreating this setup; their reviewed sync plan remains under the run directory.

The **home machine** runs the enabled user service
`natlang-dgx-development-data-sync.service`. Its plan and status are in
`runs/dgx-development-sync-20261003/`. After each complete copy it waits five
minutes, then repeats. A failed transfer is logged and retried. The first full
copy can take considerably longer; inspect the status before assuming the
entire pipeline is present. A priority copy supplies the current recipe input
closure and corpus outputs first; the broad transfer waits for its hash audit,
so two transfers do not write the same data at once. The enabled
`natlang-dgx-development-priority-sync.service` completes or retries the initial
copy after a restart, then verifies the exact 81 recipe input hashes. That proof
does not mark the larger mirror complete; its completion is recorded separately
in `sync-status.json`.

```bash
# Run on the home machine.
systemctl --user status natlang-dgx-development-data-sync.service
cat runs/dgx-development-sync-20261003/sync-status.json
tail -n 25 runs/dgx-development-sync-20261003/push.log
```

Home owns the mirrored corpus and imported campaign paths. DGX development
generation should write to **`runs/dgx-development-generated/<unique-run>/`**.
That namespace is pulled back to the same home-relative path before each push
and is excluded from pushes. Keep experiments under unique names; do not write
new development results into a mirrored production campaign directory.

Production DGX campaigns under `~/natlang-remote` continue through the existing
exact-assignment importer. The resulting local exports and raw evidence are
then replicated into this checkout. This preserves model/runtime/source
provenance and keeps failed jobs distinct from complete results.

The data service does not synchronize source code, delete files, or change
training/generation service authority. Destination revisions replaced by a
home mirror update are preserved outside the checkout under
`/mnt/external/natlang-development-data/.sync-history/`. Files are staged and
renamed rather than updated in place. Pull-side replaced revisions are also
retained locally. The service checks the external mount/device, real tracked
parent directories, reviewed root allowlists and free-space reserve each pass.

Changes to canonical source datasets should be prepared in a development run
and reviewed before replacing a published source. Current ready manifests,
source exclusions and protected heldout splits still apply on both machines.
