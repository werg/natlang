# Natlang skills

This repository includes two standalone skills for coding agents (Claude Code, Codex):

| Skill | Use |
|---|---|
| [natlang-authoring](natlang-authoring/SKILL.md) | Write, check, run, test and debug natlang programs; tune or optimize them |
| [natlang-integration](natlang-integration/SKILL.md) | Build Node and browser applications on the natlang runtime |

## Install

```sh
natlang skills install            # into ~/.claude/skills and/or ~/.codex/skills, whichever exist
natlang skills install --dir DIR  # or --claude, --codex
```

A reinstall replaces each skill folder whole. Restart or refresh the agent afterwards. `natlang skills --json`
prints where the skills of this installation live. Programs created with `natlang init` carry an `AGENTS.md` that
points to the authoring skill.

The skills describe the TypeScript runtime, compiler and CLI. `@natlang/node` and `@natlang/browser` are built from
this repository; see [native packages and executables](../NATIVE_PACKAGES.md). Links from the skills into `spec/` and
`docs/` resolve in a repository checkout; an installed copy works without them.

## Use

- “Use `$natlang-authoring` to write a natlang function that triages support tickets.”
- “Use `$natlang-integration` to embed the browser runtime in this app.”

The authoring skill's [review example](natlang-authoring/assets/review/) is a named function with a callable folder
holding a natural-language helper and an exact one; `ts-host/test/distributed-skills.test.mjs` runs it with a scripted
model.

## What belongs here

The skills are for agents writing and embedding natlang programs. Engineering notes about this repository's own
training applications and native training capture live in `docs/SEMANTIC_APPLICATION_NOTES.md` and
`docs/NATIVE_RUNTIME_NOTES.md`; keep them out of the skills. A language or runtime change updates the reference file
that covers it (`references/*.md`), with recovery advice for a new user-facing error; `SKILL.md` stays the short
working guide.
