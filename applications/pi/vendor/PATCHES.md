# Patches to the vendored pi-durable

`vendor/durable` is pi-durable from https://github.com/earendil-works/pi at f10993b (`packages/durable`: src, test,
docs; MIT, see LICENSE). It builds against the npm releases of `@earendil-works/chord` and `@earendil-works/pi-ai`
(applications/pi/package.json). Its own suite runs with `npx vitest --run` in this directory; three test files import
other packages of pi's monorepo and do not load here (provider-session-cache-e2e, session-states,
system-order-cache-e2e).

Changes, each kept minimal:

1. Exports for the port's host (no behavior change): `appendAssistant` and `streamResponse` (harness/generation.ts);
   `fromSlot`, `toolDiagnostic`, `truncated`, `boundContent`, `publishProgress` and the `Reported` type
   (harness/tool.ts).
