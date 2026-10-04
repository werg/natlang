# Static browser evaluator pilot

This is measurement infrastructure, not approval to collect or admit training data.
The host owns reference measurements and rewards. Containers receive candidate HTML
and an unpredictable protocol token only. JavaScript and service workers are disabled,
network requests are blocked twice (Docker and browser routing), and no host directory,
reference, provider key or GPU is mounted. Rendering uses one CPU and bounded memory,
processes, output and wall time. Allocation failures remain unscored.

Build locally:

```sh
docker build -t natlang-static-browser:3 scripts/visual-browser
docker image inspect natlang-static-browser:3 --format '{{.Id}}'
```

Pass the immutable image ID to both audits; never use the mutable tag as an objective
identity. The multiarch base digest and Playwright npm/browser versions are pinned.
Record distinct image IDs per architecture. Support/query comparisons must use the
same image and host; these measurements make no timing or cross-host speed claims.

Chromium keeps its own sandbox, runs as `pwuser`, and uses the official Playwright
v1.63.0 seccomp profile, byte-pinned in the host scorer. `SYS_CHROOT` is the sole
retained capability: removing it produced Chromium's `sys_chroot` startup failure.
There is no privileged mode, host IPC, SYS_ADMIN, unconfined seccomp or root browser.
The profile comes from
https://github.com/microsoft/playwright/blob/v1.63.0/utils/docker/seccomp_profile.json.

The pilot preserves normalized text, static semantic affordances, desktop text
colors/backgrounds/size and rough geometry. Its reward measures worst-viewport
overflow, clipping and legibility. It does **not** verify JS behavior, accessibility,
complete aesthetic fidelity, screenshot reconstruction or arbitrary paint effects.
Generated pseudo-element content, masks, filters, clipping tricks and ambiguous
source visibility are unsupported; source cases remain held instead of being
scored as model failures. Never silently relax those checks to raise yield.

```sh
node ts-host/scripts/skills/audit-visual-reward.mjs IMAGE_ID fresh-mutations.json
node ts-host/scripts/skills/audit-visual-browser.mjs \
  --tasks data/self-improvement/visual-frontend/intake-v3/websight-v01/artifact-source-tasks.jsonl \
  --image IMAGE_ID --out fresh-screen-directory
```

Do not rebuild scorer code while an audit is running. The source screen records a
before/after scorer hash check. Failed and superseded pilots remain in `runs/`.
Before native episode registration: complete paint/design mutation review, adjudicate
the topic/template grouping proposal, enforce globally closed support/query groups,
and bind image identity into collection, replay and admission. Cases without baseline
headroom do not become improvement examples. There are no model calls in these audits.
