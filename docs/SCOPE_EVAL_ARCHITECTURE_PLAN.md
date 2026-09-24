# Typed TypeScript scope and reducer architecture

Status: clean-break architecture implemented in the canonical TypeScript host;
this document records its design contract. [`spec/SPEC.md`](../spec/SPEC.md) is
normative when details differ.

## Source model

A natlang program is a set of typed functions. Crisp helpers are ordinary
TypeScript functions, normally written as default exports. Natural-language
functions declare typed parameters and a return type in frontmatter and
contain instructions executed line by line. Both kinds of imported function
use the same names and positional calls. Synchronous TypeScript functions
return directly; asynchronous TypeScript and natural-language functions return
promises.

Authored signatures use standard TypeScript data types: `string`, `number`,
`boolean`, `null`, records, arrays, `Record<string, T>`, named aliases, and
literal unions. The runtime validates inputs and results structurally at
function boundaries, after eval transactions, and before a natural-language
function completes.

## Model surface

Natural-language functions run in a persistent TypeScript scope. Their core
execution tools are:

- `eval(code)` for ordinary TypeScript declarations, expressions, control
  flow, and imported function calls;
- `read_page(id, page)` for reading output a tool result cut off;
- `compact_history(note)` for shortening a long conversation (older outputs stay in `transcript`);
- `return_result(status, value?, reason?)` for returning a typed value and finishing, or for explicit
  `blocked` and `failed` exits.

Normal lambdas can inspect, edit, and diff imported functions through function
tools. They do not receive model-facing filesystem tools. A function completes
when a typed eval result exists, every substantive instruction line is closed,
and no child operation remains unresolved. A compatible final expression or
assignment to `result` supplies the typed value. There is no separate done action.

## Directory reducers and files

A directory reducer receives an isolated writable copy of its input folder.
Its file tools and folder filesystem API use paths relative to that folder,
without a synthetic root prefix. A reducer may create, edit, move, and remove
files. The fixed function set remains separate and cannot be structurally
changed through file operations.

An awaited direct call, `await reducer(folder, ...args)`, returns the typed value
and discards the reducer's folder changes. `await folder.apply(reducer, ...args)`
forks the folder, runs the same reducer, and retains its selected changes.
Nested reducer calls follow the same rule: only an explicit `folder.apply`
retains a patch in the enclosing folder.

Folder writes are serialized. A reducer failure leaves the input folder
unchanged. A successful result can select all changes or use `commit` with
relative include and exclude patterns. Ordinary functions can use native
filesystem capabilities through explicitly imported host helpers, but they
cannot browse an ambient filesystem through the model surface.

## Execution and continuation

Eval compiles TypeScript snippets against the persistent scope and runs them
through the canonical TypeScript host. Each eval transaction stages variable
and parameter changes; compile, execution, host-call, or type failures do not
commit those changes. Eval output is visible to the model, and a compatible
value can supply or replace the function result until completion.

The runtime serializes typed locals, the current result, suspended
children, folder revisions, and effect observations for continuation. It does
not rely on old conversation text to reconstruct completed work. External host
effects remain application-owned: a failed acknowledgement may leave an
unknown outcome, so retries need stable operation identities and preserved
receipts.

## Implementation and compatibility

Node and browser TypeScript hosts implement this source contract. Provider
adapters may serialize the exposed tools differently; that transport syntax
does not alter natlang source. Python code remains available for offline
compatibility and historical trace work and is not the canonical runtime or
teacher collection path.

Training examples should reflect ordinary TypeScript imports and loops,
`eval` with top-level `return`, `return_result`, function tools for function source, and relative
filesystem operations that occur only in directory reducers. Legacy path
binding and call-combinator traces may remain as historical data only when
labelled as such; they are not examples of the current interface.
