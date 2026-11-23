# Idiom Reviewer (`applications/idiom-reviewer/`): Decomposition

Tier 0 flagship tool (WS3 I2, WS6 §2; plans/semantic-fp/03 §3.1). The crisp floor of `natlang check`
(`ts-host/src/compiler/idiom.ts`) reports what is exact. This reviewer judges what is not: whether prose in a `.nl` body
is judgment or mechanism.

`natlang run applications/idiom-reviewer -- DIR` reviews every `.nl` under DIR and prints an `IdiomReport`; exit 1 when
it has findings. `natlang check --idiom` (plan 03 §3.1) does not call it yet.

## Parts

| Part | Form | Why |
|---|---|---|
| Read a program's `.nl` sources and whether each has TypeScript helpers | crisp (`main.ts` `programFiles`) | file system |
| Split a source into front matter and steps (numbered items, else paragraphs) | crisp (`splitSource`, `stepsOf`) | exact text structure; same code as the migrator |
| What each step asks for: `judgment`, `mechanism`, `guard-for-type`, `redundant` | fn `stepKind.nl` (`readout: decision`), mapped with `semanticMap`, `escalate` when unsure | judgment; the migrator's `stepKind.nl` contract |
| Is the function as a whole a prose interpreter? | fn `isProseInterpreter.nl` (`readout: decision`) over the kinded steps and `hasHelpers` | judgment over the whole function, not a count threshold |
| Does the body route on trigger words instead of meaning? | fn `detectCueWords.nl` (`readout: decision`) | judgment |
| Is a parameter external text typed as trusted? | fn `detectUntypedExternal.nl` (`readout: decision`) per parameter not already `Untrusted<...>` | judgment; reading `args:` is crisp (`argsOf`) |
| Finding per decision, with its suggestion | crisp | fixed text per finding kind |

## One implementation

`stepKind.nl`, `splitSource` and `stepsOf` are copies of `applications/migration/program`'s, because a package cannot
import another package's sources yet. Both tools should call one shared function (plan 03 §3.1); its home is
`@natlang/std`.
