# Natlang skills

This repository includes two standalone skills for coding agents:

| Skill | Use |
|---|---|
| [natlang-authoring](natlang-authoring/SKILL.md) | Create and improve TypeScript and natural-language functions |
| [natlang-integration](natlang-integration/SKILL.md) | Embed the Node or browser runtime in an application |

## Install

Copy the complete skill directories into the agent's skill discovery folder.
Keep the references, metadata, and examples alongside each `SKILL.md`.
For Codex, the destination is commonly `~/.codex/skills/`:

```sh
cp -R skills/natlang-authoring ~/.codex/skills/
cp -R skills/natlang-integration ~/.codex/skills/
```

Restart or refresh the agent's skill discovery after copying. To update a
skill, compare the installed folder with the version in this checkout and copy
the complete updated folder.

The skills describe the native TypeScript runtime. `@natlang/node` and
`@natlang/browser` are built from this repository; see [native packages and
executables](../NATIVE_PACKAGES.md) for package entry points and build steps.

## Use

- “Use `$natlang-authoring` to inspect and improve this natlang codebase.”
- “Use `$natlang-integration` to embed the browser runtime in this app.”

The authoring reference includes a small multi-file example under
`natlang-authoring/assets/review/`. Run examples through the TypeScript host and
check real model behavior separately from scripted runtime plumbing.
