# Native TypeScript authored roots

A bounded reference with the actual e6 vendored Core, Vibe ProgramLoader and
SQLite CallStore returned the correct typed value from a TS entry, but retained
only its NL child as a root. The TS entry's output and parent identity were absent.
This was not a model or provider error.

Mechanism inventory:

- `runtime/virtual-project.ts` public `compileVirtualProject.require` exposed raw
  emitted exports. ProgramLoader `packages/natlang/src/programs.ts:258` and browser
  playground `src/app/playground.ts:135` consume that API. Internal emitted module
  imports continue to use the same original module instances and exports.
- `runtime/callable.ts` callable-folder TS exports also invoked raw native functions.
- `runtime/kernel.ts:890` records NL invocation inputs, outputs and child custody;
  `node-runtime.ts:78` separately records standalone native interpreter roots.
  Neither is entered when a host calls an ordinary compiled TS function.
- `runtime/native-entry.ts` now records the actual host entry at both exposed TS
  invocation boundaries, preserving synchronous/asynchronous values and errors.
  It establishes the true parent frame for NL descendants, records native service
  effects and exact typed inputs/output, and closes actual child custody. Helpers
  inside an already recorded NL/native invocation retain existing behavior.
- The mechanism is shared by node, browser and workerd Core. Managed llama,
  in-page wllama, Python native and SGLang are model drivers, not alternate TS
  module execution implementations (docs/ENGINE_PARITY.md); their inference
  protocols are unchanged. Native root tracing does not call a model.
- Vibe RecordsStore `record` adds captured exchanges to real Core records and
  assigns the owner/program root; the driver capture API owns those exchanges.
  Core CallStore alone does not attach Vibe `model_turns`. No synthetic root or
  NL forwarding wrapper may substitute for this actual native record.
- Exporters/readers already accept crisp roots with actual descendant decisions:
  Host951e03fe and natlang b56feaeee retain actual root-output/evidence bindings.
  These do not require a generation at the authored root.

Default: native authored root recording is enabled when a call store exists.
`recording.nativeRoots=false` opts out. Unconstrained already compiled TS remains
valid: signature metadata uses declared types when present and unknown otherwise;
parsing metadata does not impose the callable-folder constrained source policy on
an already accepted compiler product. Re-export accessors remain live.

Reference fixtures use scripted model replies only and are not training data.
The frozen corpus release is unchanged; this Core fix can enter a later vendor
update independently of collection from existing NL entries.
