# Natlang skills for coding agents

Use these skills when building with natlang. They are for capable authors and integrators programming a potentially much smaller model interpreter. They cover complete algorithms and real host/application wiring, including long runs, native objects, generated UIs, and semantic evaluation.

| Skill | Use it for |
|---|---|
| [natlang-authoring](natlang-authoring/SKILL.md) | Typed `.nl` codebases, crisp helpers, semantic algorithms, explicit control flow, long-run state, tests and training evidence |
| [natlang-integration](natlang-integration/SKILL.md) | Python/Node/browser embedding, model adapters, native frontend reducers, host authority, effects, streams, persistence and recovery |

Each skill includes its own references and can be installed alone. Authoring includes a runnable multi-file review example. Core instructions are short; load detailed references only as needed. The format follows the portable [Agent Skills specification](https://agentskills.io/specification); optional `agents/openai.yaml` supplies discovery UI metadata.

## Install

From an installed natlang Python package or an editable checkout:

```bash
python -m natlang.install_skills --list
python -m natlang.install_skills --dest ~/.codex/skills
# Or install only the authoring skill:
python -m natlang.install_skills natlang-authoring --dest /path/to/agent/skills
```

Use the Python environment where natlang is installed (`.venv/bin/python` in this repository). The destination is explicit; the installer never changes runtime/model settings or overwrites an existing skill. For an update, compare the installed copy with the new bundled version, then move the old directory aside and install again. Keep skills aligned with the natlang version in use.

Agents with another discovery location can receive the same complete folders through `--dest`, or by copying them manually. Node-only users can copy `skills/natlang-authoring` and `skills/natlang-integration` from this checkout; Python is not required to use the skill text. Do not copy only `SKILL.md`: references, metadata and examples belong with it. Reload your agent's skill discovery according to its client. An agent that cannot install skills can open the entrypoints directly from the repository.

These skills ship in the Git source and Python wheel. The current private TypeScript package is installed from a built local checkout; its npm artifact does not bundle these directories. Use this repository's skills alongside it. No public registry or model-weight availability is implied.

## Invoke

Examples:

- “Use $natlang-authoring to implement a semantic document merger that preserves competing intentions.”
- “Use $natlang-integration to embed a natlang reducer and generated interface in this browser app.”
- “Use both natlang skills to build a notebook whose dependency traversal is driven by natlang.”

Descriptions support automatic discovery; explicit invocation is optional. These skills do not confer authority to restart shared inference servers, execute external effects, or change application scope.

## Maintenance and verification

The references were researched against the repository's loaders, host APIs, runtime tests, and the public skill format. They distinguish implemented behavior from design proposals and historical experiments. Important source anchors are listed in each reference. When changing a public natlang contract, update the relevant skill, current documentation, and executable example/tests together.

`tests/test_distributed_skills.py` tests relocation, installation conflicts, bundled references, and the example through the Python runtime. `ts-host/test/distributed-skills.test.mjs` runs that same source through the native host with scripted model turns. These tests establish wiring and exact aggregation, not semantic model quality. Validate frontmatter with your client's skill validator or `skills-ref validate skills/natlang-authoring` and `skills-ref validate skills/natlang-integration` when that tool is installed.
