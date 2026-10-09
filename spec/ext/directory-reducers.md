# Directory reducers

Extension version **0.5-draft**, extending the [core specification](../SPEC.md) (0.5-draft). An extension is optional: a program or host that does not use it is unaffected.

**Scope.** Functions whose first parameter is a `Folder`: how they get an isolated writable copy, how changes are committed with `folder.apply`, and how a reducer computes a new context revision.

A directory reducer's first parameter is a `Folder` (or a handle from
`folder.dir(path)`), available as `folder` in eval. It works on an isolated
writable copy; paths are relative to the folder. `await reducer(folder, ...args)`
returns the typed result and discards file changes.
`await folder.apply(reducer, ...args)` retains the committed changes. A typed
result selects every change; `commit` selects changes by glob. Folder writers
serialize.

A reducer over a context folder is the way to compute a new context: its staged
tree, once compiled, is a file context from which new executable nodes may be
bound (see Contexts). Self-improvement is a function from a context and evidence
to a new context revision; promotion binds a program to that revision, or
`folder.apply` commits it to a real directory.
