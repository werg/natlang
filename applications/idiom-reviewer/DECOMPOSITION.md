# Idiom Reviewer (`applications/idiom-reviewer/`): Decomposition

Tier 0 flagship tool (WS3 I2, WS6 §2; plans/semantic-fp/03 §3.1). The crisp floor of `natlang check`
(`ts-host/src/compiler/idiom.ts`) reports what is exact. This reviewer judges what is not: whether prose in a `.nl` body
is judgment or mechanism.

`natlang run applications/idiom-reviewer -- DIR` reviews every `.nl` under DIR and prints an `IdiomReport`; exit 1 when
it has findings. `natlang check --idiom` invokes this reviewer over the checked project. Sibling `types.ts` definitions are
resolved transitively so records containing `Untrusted<...>` fields (such as `Hunk`, `IncomingMessage`, `FailureTrace`)
are recognized as already fenced.

## Parts

| Part | Form | Why |
|---|---|---|
| Read a program's `.nl` sources and whether each has TypeScript helpers | crisp (`main.ts` `programFiles`) | file system |
| Split a source into front matter and steps (numbered items, else paragraphs) | crisp (`@natlang/std` `instructionsOf`) | exact text structure; one implementation with the migrator |
| What each step asks for: `judgment`, `mechanism`, `guard-for-type`, `redundant` | fn `@natlang/std` `stepKind` (`readout: decision`), mapped with `semanticMap`, `escalate` when unsure | judgment; shared with the migrator |
| Is the function as a whole a prose interpreter? | fn `isProseInterpreter.nl` (`readout: decision`) over the kinded steps and `hasHelpers` | judgment over the whole function, not a count threshold |
| Does the body route on trigger words instead of meaning? | fn `detectCueWords.nl` (`readout: decision`) | judgment |
| Is a parameter external text typed as trusted? | fn `detectUntypedExternal.nl` (`readout: decision`) per parameter not already `Untrusted<...>` | judgment; reading `args:` is crisp (`argsOf`) |
| Finding per decision, with its suggestion | crisp | fixed text per finding kind |

## One implementation

The step judgment and the step splitter are `@natlang/std`'s (`stepKind`, `instructionsOf`, `stepsOf`;
ts-host/std/stepKind.nl), shared with the program migrator (`applications/migration/program`). Their examples moved
with them (`ts-host/std/examples.json`).
