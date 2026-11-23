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

## Declared files

`writes:` in a directory reducer's frontmatter declares the files its calls write, as one type: a record whose field
names are paths from the folder, or globs (`*` is any name within one folder level, `**` any number of levels), and
whose field types type each file's text (`string`, a string literal, or a union of them). Usually it names an alias of
`types.ts`, whose doc comments say what each file holds (line templates, separators, number formats, the empty case,
order) and, on the alias, what holds for all of them:

```ts
/** Every line of a written file ends with a newline. */
export type DeskFiles = {
  /** One line per applied request, `<request id> | <change>`, in request file name order; `none` when none applied. */
  "log/applied.md": string;
  /** The applied requests, moved here under their own names. */
  "done/*.md": string;
};
```

```yaml
kind: directory-reducer
returns: number
writes: DeskFiles
```

A call's opening shows the declaration under its signature (`The files it writes: DeskFiles`) with the aliases it
names, doc comments included, as it shows the aliases of the signature. A doc comment on an alias in `types.ts` is
shown above it. When the call returns success, every file it created or changed must match a field and its text must
fit that field's type; otherwise the return is refused with `writes-unsatisfied` and one sentence naming the file
(`notes.md is not among the files tally writes (tally.md, done/*.md); undo that change ...`), and the call repairs it.
Files it removed (the source of a move) and files it left alone are not checked. A function that is not a directory
reducer, a type that is not a record, a field type that is not text, and a field name that leaves the folder are load
errors. An edit cannot change the `writes:` line (`protected`).
