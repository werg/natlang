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

A directory reducer need not be a `.nl` function. `folder.apply`, `folder.propose` and
`folder.iterateOn` also take a host or TypeScript function `(folder, ...args)`. It has
the same contract: it runs on an isolated writable copy of the folder, receives that
copy as `folder`, may call `folder.apply` on natural-language reducers inside it, and
its changes are validated and installed when it returns and discarded when it throws.
Use one for the crisp mechanism around natural-language stages (a fixed sequence of
checks and calls), so that no model call sequences it. A natural-language function that
is not a `kind: directory-reducer` is not accepted as a reducer.

A reducer over a context folder is the way to compute a new context: its staged
tree, once compiled, is a file context from which new executable nodes may be
bound (see Contexts). Self-improvement is a function from a context and evidence
to a new context revision; promotion binds a program to that revision, or
`folder.apply` commits it to a real directory.
