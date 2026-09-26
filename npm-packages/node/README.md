# `@natlang/node`

The Node.js runtime and compiler for natlang: `nl`, `iterateOn`,
`createNatlangRuntime`, named `.nl` loading, `buildProject` / `checkProject`,
folders, services, traces, `EventLoop` and terminal utilities, the managed local
model session, Pi provider sessions, and `.nlpkg` package APIs. Install
`@natlang/cli` for the `natlang` command.

The model API discovers and validates explicit, natlang-managed, and PATH
llama.cpp runtimes. Sessions stay lazy until `prepare()` or the first model turn.

`createManagedModelSession(profile)` accepts the same model profile fields as
the CLI: neither `endpoint` nor `provider` for natlang's managed local model,
`endpoint` for an OpenAI-compatible service, or `provider` and `model` for Pi.
Use `session.turn` as a `ModelDriver`, and close the session after the runtime:

```ts
import { createManagedModelSession, createNatlangRuntime } from '@natlang/node';

const session = createManagedModelSession({ provider: 'anthropic', model: 'MODEL_ID' });
const runtime = createNatlangRuntime({ model: request => session.turn(request) });
try {
  await session.prepare();
  // await runtime.run(() => ...);
} finally {
  runtime.close();
  await session.close();
}
```

`loadModelConfiguration`, `resolveModelChoice`, and `createResolvedModelSession`
expose the CLI's selection and validation for applications that need the same
precedence. The `@natlang/node/model/pi` subpath exposes the lower-level Pi
backend for programmatic provider hooks. See the repository's
[model configuration guide](https://github.com/werg/natlang/blob/main/NATIVE_PACKAGES.md#authority-and-the-model-runtime)
for profile fields, credentials, and option behavior.
