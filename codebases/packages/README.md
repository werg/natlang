# Offline natlang packages

`resolve.nl` asks the exact offline solver for compatible locks, then uses
natlang to select one by the user's purpose. `finalize.ts` rejects an invented
ID. `install.nl` validates the pinned source graph before asking the host to
publish it. Package definitions remain ordinary natlang checked definitions;
the package manager does not add an implicit import or dynamic call rule to the
runtime.

`PackageRegistry` stores cloned manifests, computes content digests, solves
exact and caret version constraints, enforces one version of each dependency,
rejects cycles and engine mismatches, and assembles namespaced definitions.
`NativeSourceWorkspace` checks type names, source links and recursion before
installation. The installer writes a complete hidden bundle, then publishes
one symlink atomically without replacing an existing target. Loading verifies
the installed bundle against the registry and reconstructs the checked source
workspace. A running lambda keeps its original source revision.

The integration test resolves two compatible root versions with a transitive
dependency, installs the latest, runs its exported natlang function, and checks
conflicts, cycles, engine mismatch, tampered locks, path traversal, existing
targets and changed installed content. It uses a scripted model callback;
semantic upgrade advice remains unmeasured.

This first registry is local and immutable. It does not fetch archives, run
lifecycle scripts, overwrite installations or silently grant a package native
host access. Its supported range grammar is exact, `*`, and caret. Upgrade
planning, cross-host bundle reproduction, source licensing and remote
publication are later product gates; none requires a new core primitive.
