# check-main linked-worktree dependency resolution

## Purpose

A full check may run from a linked git worktree whose source differs from the
canonical checkout. Its tests, build scripts and installed CLI must resolve
Natlang workspace packages from that tested source tree. Third-party packages
can continue to come from the canonical checkout's already-installed
`node_modules` trees.

## Data and stages

1. Read the worktree root, common git checkout root, and the root package's
   declared npm workspace paths.
2. If both roots are the same, leave the existing install untouched.
3. If the worktree root `node_modules` is absent or is the exact symlink the
   old setup made to the canonical root `node_modules`, create a local
   directory. Link each canonical root dependency into it except workspace
   package entries, then link each declared workspace package from the tested
   checkout under its package name. Preserve unrelated scoped packages.
4. If a worktree root install already exists as a directory and its workspace
   links resolve locally, leave it unchanged. If it contains a foreign or
   incomplete workspace link, fail with a precise message rather than
   replacing an unknown dependency tree.
5. Continue sharing the existing `ts-host/node_modules` and per-application
   dependency directories; these contain third-party packages, while package
   resolution for workspaces ascends to the now-local root overlay.

## Crisp boundary

This is filesystem and npm package-resolution plumbing: exact symlink targets,
workspace names and failure behavior require deterministic code. It does not
choose versions, install packages, mutate dependency manifests, or alter model
behavior. No package installation or source-tree copying is performed.

## Verification

A temporary two-checkout test will prove that the linked worktree resolves its
own workspace package while resolving a third-party dependency from the shared
canonical install. It will also check idempotent setup and preservation of an
unrelated pre-existing `node_modules` directory. The actual stale-worktree
package adaptation test will be rerun against its own worktree package tree.
